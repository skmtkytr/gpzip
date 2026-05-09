//! Real GPU backend (wgpu). LZ77 match-finding runs on the GPU; the host
//! does the greedy walk, fixed Huffman encode (RFC 1951 §3.2.6), and gzip
//! framing (RFC 1952). Output is a standard gzip stream — verifiable by
//! `gzip -t`, `flate2`, etc.

use std::io::Write;
use std::sync::Arc;

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
            // 512 KiB: small enough that oldest-wins hash references stay
            // recent on repetitive data, big enough that the GPU has real
            // work to chew on per chunk.
            chunk_size: 512 * 1024,
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
    pub fn gzip_chunk_fn(&self) -> ChunkFn {
        let batched = Arc::clone(&self.batched);
        Arc::new(move |bytes: &[u8]| -> std::io::Result<Vec<u8>> {
            let raw = batched.submit(bytes.to_vec());
            let walked = lz77::greedy_walk(&raw, bytes);
            let deflate = deflate::encode_block_fast(&walked)?;
            Ok(deflate::gzip_wrap(&deflate, bytes))
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

impl GpuGzipCompressor {
    fn chunk_fn_for_writer(&self) -> ChunkFn {
        let batched = Arc::clone(&self.batched);
        Arc::new(move |bytes: &[u8]| -> std::io::Result<Vec<u8>> {
            let raw = batched.submit(bytes.to_vec());
            let walked = lz77::greedy_walk(&raw, bytes);
            let deflate = deflate::encode_block_fast(&walked)?;
            Ok(deflate::gzip_wrap(&deflate, bytes))
        })
    }
}
