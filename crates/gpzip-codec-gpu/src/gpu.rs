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

mod chunk;
mod context;
mod deflate;
mod identity;
mod lz77;

pub use context::GpuContext;

/// GPU codec. Holds an initialized wgpu device + queue and a precompiled
/// LZ77 pipeline. Cheap to clone (Arc'd internally).
#[derive(Clone)]
pub struct GpuBackend {
    ctx: Arc<GpuContext>,
    lz77: Arc<lz77::Lz77Pipeline>,
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
        let lz77 = Arc::new(lz77::Lz77Pipeline::new(Arc::clone(&ctx)));
        Ok(Self {
            ctx,
            lz77,
            // Smaller default than CPU because the brute-force LZ77 shader is
            // O(window) per byte. Tuned upward once the hash-table variant
            // (A-2d) is in.
            chunk_size: 256 * 1024,
            // GPU work is already serialized on the device queue; running
            // many chunks concurrently from rayon would just contend. Keep
            // two in-flight to overlap GPU compute with CPU-side encoding.
            max_in_flight: 2,
        })
    }

    pub fn context(&self) -> &GpuContext {
        &self.ctx
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
                lz77: Arc::clone(&self.lz77),
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
    lz77: Arc<lz77::Lz77Pipeline>,
    chunk_size: usize,
    max_in_flight: usize,
}

impl Compressor for GpuGzipCompressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Gzip
    }

    fn wrap_writer(self: Box<Self>, w: Box<dyn Write + Send>) -> Box<dyn Write + Send> {
        let lz77 = Arc::clone(&self.lz77);
        let chunk_fn: ChunkFn = Arc::new(move |bytes: &[u8]| -> std::io::Result<Vec<u8>> {
            // GPU: per-position LZ77 match-find.
            let raw = lz77.match_find(bytes, lz77::DEFAULT_WINDOW);
            // CPU: greedy + lazy selection, fixed-Huffman bitstream, gzip frame.
            let walked = lz77::greedy_walk(&raw, bytes);
            let deflate = deflate::encode_block(&walked)?;
            Ok(deflate::gzip_wrap(&deflate, bytes))
        });
        Box::new(ParallelChunkedWriter::new(
            w,
            self.chunk_size,
            self.max_in_flight,
            chunk_fn,
        ))
    }
}
