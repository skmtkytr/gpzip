//! Hybrid CPU+GPU codec backend.
//!
//! For algorithms supported by both backends (today: only Gzip), every
//! incoming chunk goes to whichever device is free first. Implementation
//! is a tiny lock-free semaphore: each chunk closure tries to acquire one
//! of N GPU permits with `compare_exchange`. If it gets one, it runs the
//! GPU chunk_fn; if not, it falls through to the CPU chunk_fn. Both paths
//! produce a self-contained gzip member, so the parallel writer just
//! concatenates results in order — same output contract as either backend
//! alone.
//!
//! Net effect: aggregate throughput is roughly `cpu_speed + gpu_speed`,
//! which is the only realistic way for our GPU path (slower per chunk
//! than the CPU path) to *help* end-to-end wall time.

use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use gpzip_codec_cpu::{ChunkFn, CpuBackend, GzipCompressor, ParallelChunkedWriter, ZstdCompressor};
use gpzip_codec_gpu::LazyGpuBackend;
use gpzip_core::{
    Algorithm, Capability, CodecBackend, Compressor, Decompressor, Error, Level, Result,
};

pub struct HybridBackend {
    cpu: Arc<CpuBackend>,
    gpu: Arc<LazyGpuBackend>,
}

impl HybridBackend {
    pub const NAME: &'static str = "hybrid";

    pub fn new(cpu: Arc<CpuBackend>, gpu: Arc<LazyGpuBackend>) -> Self {
        Self { cpu, gpu }
    }
}

impl CodecBackend for HybridBackend {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn supports(&self, algo: Algorithm) -> Capability {
        // Hybrid contributes nothing the CPU backend doesn't already cover;
        // the value is the *compressor*, not the capability set.
        self.cpu.supports(algo)
    }

    fn compressor(&self, algo: Algorithm, level: Level) -> Result<Box<dyn Compressor>> {
        match algo {
            Algorithm::Gzip => Ok(Box::new(HybridGzipCompressor {
                cpu_chunk_fn: GzipCompressor::chunk_fn(level),
                gpu: Arc::clone(&self.gpu),
                chunk_size: self.cpu.chunk_size(),
                cpu_workers: self.cpu.max_in_flight(),
                // 2 permits keeps 80-90% of work on the better-compressing
                // CPU path; the GPU mostly contributes during bursty
                // periods when CPU permits are saturated.
                // 1 permit minimises the GPU's drag on hybrid output: real-
                // workload measurements show the GPU path produces 4–15%
                // worse compression than CPU, so every chunk that lands on
                // the GPU costs ratio. 0 would skip GPU entirely and be
                // fastest, but hybrid keeps a single permit so a future
                // GPU improvement (or different hardware) lets the device
                // start contributing without a CLI change. Real-world
                // measurement showed gpu_workers ∈ {1,2,4,8} are all
                // slower AND worse-compressing than --backend cpu on this
                // box (Ryzen 7800X3D + RTX 4090) — see commit message for
                // the table. --backend cpu is the recommended fast path
                // until the GPU pipeline catches up.
                gpu_workers: 1,
            })),
            // Zstd has no GPU implementation today; fall through to CPU.
            Algorithm::Zstd => Ok(Box::new(ZstdCompressor::new(
                level,
                self.cpu.chunk_size(),
                self.cpu.max_in_flight(),
            ))),
            _ => self.cpu.compressor(algo, level),
        }
    }

    fn decompressor(&self, algo: Algorithm) -> Result<Box<dyn Decompressor>> {
        // GPU has no decompressor; always CPU.
        self.cpu.decompressor(algo)
    }
}

struct HybridGzipCompressor {
    cpu_chunk_fn: ChunkFn,
    gpu: Arc<LazyGpuBackend>,
    chunk_size: usize,
    cpu_workers: usize,
    gpu_workers: usize,
}

impl Compressor for HybridGzipCompressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Gzip
    }

    fn wrap_writer(self: Box<Self>, w: Box<dyn Write + Send>) -> Box<dyn Write + Send> {
        let permits = Arc::new(AtomicUsize::new(self.gpu_workers));
        let cpu_fn = self.cpu_chunk_fn.clone();
        let gpu = Arc::clone(&self.gpu);
        // Cache the materialised GPU chunk_fn after the first lookup; the
        // OnceLock also serialises concurrent first-callers so wgpu init
        // (~200 ms) runs exactly once. If init fails (no adapter), the cell
        // holds None and every subsequent chunk falls through to the CPU
        // path with no further retries.
        let gpu_fn_cell: Arc<OnceLock<Option<ChunkFn>>> = Arc::new(OnceLock::new());
        // Warm-up: skip GPU entirely for the first N chunks so small
        // inputs (1-2 chunks) finish on CPU without ever paying the
        // ~200 ms wgpu init cost. Default chunk_size is 2 MiB so N=4
        // means inputs ≤ 8 MiB stay pure-CPU. Larger inputs pay init
        // exactly once on chunk N+1.
        const GPU_WARMUP_CHUNKS: usize = 4;
        let chunk_counter = Arc::new(AtomicUsize::new(0));
        // Short-circuit when GPU is disabled (gpu_workers=0) so we never
        // pay the wgpu init cost via get_or_init below.
        let gpu_enabled = self.gpu_workers > 0;
        let composed: ChunkFn = Arc::new(move |bytes: &[u8]| {
            let n = chunk_counter.fetch_add(1, Ordering::Relaxed);
            if !gpu_enabled || n < GPU_WARMUP_CHUNKS {
                return (cpu_fn)(bytes);
            }
            let gpu_fn = gpu_fn_cell.get_or_init(|| gpu.try_get().map(|g| g.gzip_chunk_fn()));
            match gpu_fn {
                Some(f) if try_acquire(&permits) => {
                    let result = f(bytes);
                    permits.fetch_add(1, Ordering::Release);
                    result
                }
                _ => (cpu_fn)(bytes),
            }
        });
        // Total in-flight = CPU workers + GPU permits, so neither device is
        // ever idle while the other is saturated.
        let total = self.cpu_workers + self.gpu_workers;
        Box::new(ParallelChunkedWriter::new(
            w,
            self.chunk_size,
            total,
            composed,
        ))
    }
}

/// Lock-free decrement-if-positive. Returns true if a permit was claimed
/// (caller must release with `fetch_add(1)`).
fn try_acquire(permits: &AtomicUsize) -> bool {
    let mut cur = permits.load(Ordering::Acquire);
    loop {
        if cur == 0 {
            return false;
        }
        match permits.compare_exchange_weak(cur, cur - 1, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return true,
            Err(actual) => cur = actual,
        }
    }
}

// Make Error::* available; keeps a deps-clean error path on the rare
// "neither backend supports this" case (currently unreachable).
#[allow(dead_code)]
fn _force_error_used() -> Error {
    Error::NoBackend {
        algo: Algorithm::Gzip,
    }
}
