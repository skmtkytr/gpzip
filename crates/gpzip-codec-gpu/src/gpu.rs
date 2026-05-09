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
        Ok(Self {
            ctx,
            lz77,
            batched,
            // 32 KiB: matches the DEFLATE window. Keeping chunk <= window
            // means every prior position in the chunk is potentially a
            // valid back-reference (no distance-out-of-window filter), so
            // the GPU hash chain finds matches reliably even though chain
            // entries aren't ordered by position. Larger chunks (512 KiB)
            // hide most candidates behind the distance filter and tank
            // compression ratio on repetitive data — measured with chain
            // walks up to 1024 entries it still couldn't recover.
            chunk_size: 32 * 1024,
            // Bumped from 2 → 8 so the batching worker can actually fill
            // batches when the hybrid pipeline is firing many chunks at
            // once. The MAX_BATCH inside BatchedLz77 caps actual batch
            // size at the same number.
            max_in_flight: 8,
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
        let sub_chunk = self.chunk_size;
        Arc::new(move |bytes: &[u8]| -> std::io::Result<Vec<u8>> {
            if bytes.len() <= sub_chunk {
                return encode_one(&batched, bytes);
            }
            let mut out = Vec::with_capacity(bytes.len());
            for piece in bytes.chunks(sub_chunk) {
                let member = encode_one(&batched, piece)?;
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
/// the same one-piece encode path.
fn encode_one(batched: &Arc<batch::BatchedLz77>, bytes: &[u8]) -> std::io::Result<Vec<u8>> {
    let raw = batched.submit(bytes.to_vec());
    let walked = lz77::greedy_walk(&raw, bytes);
    let deflate = deflate::encode_block_fast(&walked)?;
    Ok(deflate::gzip_wrap(&deflate, bytes))
}

impl GpuGzipCompressor {
    fn chunk_fn_for_writer(&self) -> ChunkFn {
        let batched = Arc::clone(&self.batched);
        Arc::new(move |bytes: &[u8]| -> std::io::Result<Vec<u8>> {
            // The pure --backend gpu path always feeds chunks of exactly
            // self.chunk_size, so no sub-chunking needed here.
            encode_one(&batched, bytes)
        })
    }
}
