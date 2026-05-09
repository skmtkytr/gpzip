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
use std::sync::Arc;

use gpzip_codec_cpu::{ChunkFn, CpuBackend, GzipCompressor, ParallelChunkedWriter, ZstdCompressor};
use gpzip_codec_gpu::GpuBackend;
use gpzip_core::{
    Algorithm, Capability, CodecBackend, Compressor, Decompressor, Error, Level, Result,
};

pub struct HybridBackend {
    cpu: Arc<CpuBackend>,
    gpu: Option<Arc<GpuBackend>>,
}

impl HybridBackend {
    pub const NAME: &'static str = "hybrid";

    pub fn new(cpu: Arc<CpuBackend>, gpu: Option<Arc<GpuBackend>>) -> Self {
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
        match (algo, &self.gpu) {
            (Algorithm::Gzip, Some(gpu)) => Ok(Box::new(HybridGzipCompressor {
                cpu_chunk_fn: GzipCompressor::chunk_fn(level),
                gpu_chunk_fn: gpu.gzip_chunk_fn(),
                chunk_size: self.cpu.chunk_size(),
                cpu_workers: self.cpu.max_in_flight(),
                gpu_workers: gpu.max_in_flight(),
            })),
            // Zstd has no GPU implementation today; fall through to CPU.
            (Algorithm::Zstd, _) => Ok(Box::new(ZstdCompressor::new(
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
    gpu_chunk_fn: ChunkFn,
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
        let gpu_fn = self.gpu_chunk_fn.clone();
        let composed: ChunkFn = Arc::new(move |bytes: &[u8]| {
            if try_acquire(&permits) {
                let result = (gpu_fn)(bytes);
                permits.fetch_add(1, Ordering::Release);
                result
            } else {
                (cpu_fn)(bytes)
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
