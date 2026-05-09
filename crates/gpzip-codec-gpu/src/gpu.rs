//! Real GPU backend (wgpu). LZ77 match-finding runs on the GPU; the host
//! does the greedy walk, fixed Huffman encode (RFC 1951 §3.2.6), and gzip
//! framing (RFC 1952). Output is a standard gzip stream — verifiable by
//! `gzip -t`, `flate2`, etc.

use std::io::Write;
use std::sync::{Arc, OnceLock};

use gpzip_codec_cpu::{ChunkFn, ParallelChunkedWriter};
use gpzip_core::{
    Algorithm, Capability, CodecBackend, Compressor, Decompressor, Error, Level, Result,
};

mod batch;
mod chunk;
mod context;
mod deflate;
mod huffman;
mod huffman_emit;
mod huffman_emit_v2;
mod identity;
mod lz77;
mod lz77_hash;

pub use context::GpuContext;

/// Lazy wrapper around `GpuBackend` that defers wgpu init until first use.
///
/// `GpuBackend::try_init` takes ~200 ms on a typical desktop GPU because it
/// enumerates adapters, requests a device, compiles shaders, and uploads
/// the reset blob. For small inputs (one chunk or less of compressible
/// data), the CLI may finish before any chunk would have reached the GPU
/// path — paying that init cost upfront is wasted.
///
/// `OnceLock::get_or_init` serialises concurrent first-callers, so the init
/// closure runs exactly once even when many worker threads ask at the same
/// moment.
pub struct LazyGpuBackend {
    inner: OnceLock<Option<Arc<GpuBackend>>>,
}

impl LazyGpuBackend {
    pub fn new() -> Self {
        Self {
            inner: OnceLock::new(),
        }
    }

    /// Initialise on first call; subsequent calls return the cached result.
    /// Returns `None` if no GPU adapter is available — callers should fall
    /// back to a pure-CPU path.
    pub fn try_get(&self) -> Option<&Arc<GpuBackend>> {
        self.inner
            .get_or_init(|| GpuBackend::try_init().ok().map(Arc::new))
            .as_ref()
    }
}

impl Default for LazyGpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Make LazyGpuBackend a drop-in for the registry. Capability is hardcoded
/// (must be answered before init) and matches the inner GpuBackend's
/// claims; compressor/decompressor calls trigger init via try_get and
/// delegate. For paths that never compress (extract/list), GPU init
/// never runs.
impl CodecBackend for LazyGpuBackend {
    fn name(&self) -> &'static str {
        GpuBackend::NAME
    }

    fn supports(&self, algo: Algorithm) -> Capability {
        match algo {
            Algorithm::Gzip => Capability::CompressOnly,
            _ => Capability::None,
        }
    }

    fn compressor(&self, algo: Algorithm, level: Level) -> Result<Box<dyn Compressor>> {
        match self.try_get() {
            Some(gpu) => gpu.compressor(algo, level),
            None => Err(Error::NoBackend { algo }),
        }
    }

    fn decompressor(&self, algo: Algorithm) -> Result<Box<dyn Decompressor>> {
        Err(Error::DecompressionUnsupported {
            backend: GpuBackend::NAME,
            algo,
        })
    }
}

/// GPU codec. Holds an initialized wgpu device + queue, a precompiled
/// hash-table LZ77 pipeline, and a background batching worker. Cheap to
/// clone (Arc'd internally).
#[derive(Clone)]
pub struct GpuBackend {
    ctx: Arc<GpuContext>,
    /// Held so the pipeline (and its buffer pool) stays alive as long as
    /// the batching worker — the worker holds its own Arc but a stray
    /// drop here would still surprise the cleanup order. Read by no one
    /// today; #[allow(dead_code)] documented in the field comment.
    #[allow(dead_code)]
    lz77: Arc<lz77_hash::Lz77HashPipeline>,
    batched: Arc<batch::BatchedLz77>,
    /// GPU dynamic-Huffman encoder (D-3). Held so the per-call BufferSet
    /// pool persists across chunks. Set to None to fall back to the
    /// host's `encode_block_fast` (controlled by `GPZIP_GPU_ENCODE`).
    huffman_emit: Option<Arc<huffman_emit_v2::HuffmanEmitV2Pipeline>>,
    chunk_size: usize,
    max_in_flight: usize,
}

impl GpuBackend {
    pub const NAME: &'static str = "gpu";

    /// Probe for an adapter, initialize a device, and compile the LZ77
    /// pipeline. Returns `Err` if no GPU is available so callers can fall
    /// back to CPU.
    pub fn try_init() -> Result<Self> {
        let ctx = Arc::new(
            GpuContext::try_init().map_err(|e| Error::Codec(format!("wgpu init failed: {e}")))?,
        );
        let lz77 = Arc::new(lz77_hash::Lz77HashPipeline::new(Arc::clone(&ctx)));
        let batched = Arc::new(batch::BatchedLz77::new(
            Arc::clone(&lz77),
            lz77_hash::DEFAULT_WINDOW,
        ));
        // GPU encoder is opt-in for now via GPZIP_GPU_ENCODE=1. The host
        // `encode_block_fast` is faster per single chunk on this hardware
        // (see D-3/D-4 bench in commit history); the toggle is here so the
        // production swap can be measured end-to-end without a code change.
        let huffman_emit = if std::env::var_os("GPZIP_GPU_ENCODE").is_some() {
            Some(Arc::new(huffman_emit_v2::HuffmanEmitV2Pipeline::new(
                Arc::clone(&ctx),
            )))
        } else {
            None
        };
        Ok(Self {
            ctx,
            lz77,
            batched,
            huffman_emit,
            // 256 KiB. Rationale: GPU dispatch overhead is roughly fixed
            // per BatchedLz77 submission (~150 µs poll + ~50 µs misc),
            // so per-byte cost scales with 1/chunk_size. The original
            // 32 KiB cap was set because a hash-chain design lost match
            // quality past chunk > window; the segmented-hash design we
            // settled on bounds lookup walks by window (lookup walks at
            // most window/SEG_SIZE = 8 segments back), so larger chunks
            // don't hurt match quality. The new HuffmanEmitV2 encoder's
            // single-pass scan extends to n_workgroups ≤ WG_SIZE = 1024
            // i.e. 1M tokens per chunk.
            chunk_size: 128 * 1024,
            // 16: lets ParallelChunkedWriter dispatch enough chunks for
            // BatchedLz77 to keep filling batches back-to-back. Measured
            // 5-trial averages on 64 MB workloads: 8→16 saved 6% wall on
            // rand and 4% on bin; 32 and 64 plateaued.
            max_in_flight: 16,
        })
    }

    pub fn context(&self) -> &GpuContext {
        &self.ctx
    }

    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    pub fn max_in_flight(&self) -> usize {
        self.max_in_flight
    }

    /// Per-chunk gzip-member encoder backed by the GPU's hash-table LZ77
    /// shader plus the host's dynamic-Huffman writer. Routes through the
    /// background batcher so concurrent chunks share a single GPU
    /// submission.
    ///
    /// Inputs larger than `self.chunk_size` are split into sub-chunks of
    /// that size; each sub-chunk produces an independent gzip member, and
    /// concatenated members form a valid gzip stream (RFC 1952 §2.2). This
    /// lets the GPU pipeline keep its preferred (small, ≤ window) chunk
    /// size for match-finding quality even when the parallel writer feeds
    /// larger blocks — the hybrid path in particular sends 2 MiB chunks
    /// matching the CPU configuration.
    pub fn gzip_chunk_fn(&self) -> ChunkFn {
        let batched = Arc::clone(&self.batched);
        let huffman_emit = self.huffman_emit.clone();
        let sub_chunk = self.chunk_size;
        Arc::new(move |bytes: &[u8]| -> std::io::Result<Vec<u8>> {
            if bytes.len() <= sub_chunk {
                return encode_one(&batched, huffman_emit.as_deref(), bytes);
            }
            let mut out = Vec::with_capacity(bytes.len());
            for piece in bytes.chunks(sub_chunk) {
                let member = encode_one(&batched, huffman_emit.as_deref(), piece)?;
                out.extend_from_slice(&member);
            }
            Ok(out)
        })
    }
}

impl CodecBackend for GpuBackend {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn supports(&self, algo: Algorithm) -> Capability {
        match algo {
            Algorithm::Gzip => Capability::CompressOnly,
            _ => Capability::None,
        }
    }

    fn compressor(&self, algo: Algorithm, _level: Level) -> Result<Box<dyn Compressor>> {
        match algo {
            Algorithm::Gzip => Ok(Box::new(GpuGzipCompressor {
                batched: Arc::clone(&self.batched),
                chunk_size: self.chunk_size,
                max_in_flight: self.max_in_flight,
            })),
            other => Err(Error::UnsupportedAlgorithm {
                backend: Self::NAME,
                algo: other,
            }),
        }
    }

    fn decompressor(&self, algo: Algorithm) -> Result<Box<dyn Decompressor>> {
        Err(Error::DecompressionUnsupported {
            backend: Self::NAME,
            algo,
        })
    }
}

struct GpuGzipCompressor {
    batched: Arc<batch::BatchedLz77>,
    chunk_size: usize,
    max_in_flight: usize,
}

impl Compressor for GpuGzipCompressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Gzip
    }

    fn wrap_writer(self: Box<Self>, w: Box<dyn Write + Send>) -> Box<dyn Write + Send> {
        let chunk_fn = self.chunk_fn_for_writer();
        Box::new(ParallelChunkedWriter::new(
            w,
            self.chunk_size,
            self.max_in_flight,
            chunk_fn,
        ))
    }
}

/// Free function so both `gzip_chunk_fn` and `chunk_fn_for_writer` share
/// the same one-piece encode path. When `huffman_emit` is `Some`, the
/// dynamic-Huffman block bitstream is built on the GPU (D-3); else the
/// host's `encode_block_fast` runs.
fn encode_one(
    batched: &Arc<batch::BatchedLz77>,
    huffman_emit: Option<&huffman_emit_v2::HuffmanEmitV2Pipeline>,
    bytes: &[u8],
) -> std::io::Result<Vec<u8>> {
    let raw = batched.submit(bytes.to_vec());
    let walked = lz77::greedy_walk(&raw, bytes);
    let deflate = match huffman_emit {
        Some(p) => p
            .emit_dynamic_block_v3(&walked)
            // If the GPU dynamic build fails (Huffman tree too tall —
            // extremely rare), fall back to the host fast encoder.
            .or_else(|_| deflate::encode_block_fast(&walked))?,
        None => deflate::encode_block_fast(&walked)?,
    };
    Ok(deflate::gzip_wrap(&deflate, bytes))
}

impl GpuGzipCompressor {
    fn chunk_fn_for_writer(&self) -> ChunkFn {
        let batched = Arc::clone(&self.batched);
        Arc::new(move |bytes: &[u8]| -> std::io::Result<Vec<u8>> {
            // GpuGzipCompressor doesn't currently carry the GPU encoder;
            // the env-var-controlled swap is wired through
            // `GpuBackend::gzip_chunk_fn` instead, which is what the
            // hybrid path uses. The pure --backend gpu compressor stays
            // on host encoder for now (where the production-swap
            // experiment pays off the most is in the hybrid build, since
            // it has spare CPU permits).
            encode_one(&batched, None, bytes)
        })
    }
}
