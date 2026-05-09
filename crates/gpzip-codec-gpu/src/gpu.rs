//! Real GPU backend (wgpu). A-1 stage: identity compute shader proves the
//! upload / dispatch / readback pipeline works end-to-end on the host's GPU.
//! Compression is still finalized on the CPU (gzip member encoding) — the
//! actual LZ77 and Huffman shaders land in A-2 onward.

use std::io::Write;
use std::sync::Arc;

use flate2::write::GzEncoder;
use flate2::Compression;
use gpzip_codec_cpu::{ChunkFn, ParallelChunkedWriter};
use gpzip_core::{
    Algorithm, Capability, CodecBackend, Compressor, Decompressor, Error, Level, Result,
};

mod chunk;
mod context;
mod identity;
mod lz77;

pub use context::GpuContext;

/// GPU codec. Holds an initialized wgpu device + queue and a precompiled
/// identity pipeline. Cheap to clone (Arc'd internally).
#[derive(Clone)]
pub struct GpuBackend {
    ctx: Arc<GpuContext>,
    identity: Arc<identity::IdentityPipeline>,
    chunk_size: usize,
    max_in_flight: usize,
}

impl GpuBackend {
    pub const NAME: &'static str = "gpu";

    /// Probe for an adapter, initialize a device, and compile the bring-up
    /// pipeline. Returns `Err` if no GPU is available so callers can fall
    /// back to CPU.
    pub fn try_init() -> Result<Self> {
        let ctx = Arc::new(
            GpuContext::try_init().map_err(|e| Error::Codec(format!("wgpu init failed: {e}")))?,
        );
        let identity = Arc::new(identity::IdentityPipeline::new(Arc::clone(&ctx)));
        Ok(Self {
            ctx,
            identity,
            chunk_size: 2 * 1024 * 1024,
            // GPU work is already serialized on the device queue; running many
            // chunks concurrently from rayon would just contend. Keep two
            // in-flight to overlap compute with CPU-side gzip framing.
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
            // Bring-up: chunks pass through the GPU (identity), then a CPU
            // gzip encoder frames them as a gzip member. Same output as
            // CpuBackend; the GPU step is proof-of-plumbing only.
            Algorithm::Gzip => Capability::CompressOnly,
            _ => Capability::None,
        }
    }

    fn compressor(&self, algo: Algorithm, level: Level) -> Result<Box<dyn Compressor>> {
        match algo {
            Algorithm::Gzip => Ok(Box::new(GpuGzipCompressor {
                identity: Arc::clone(&self.identity),
                level,
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
    identity: Arc<identity::IdentityPipeline>,
    level: Level,
    chunk_size: usize,
    max_in_flight: usize,
}

impl Compressor for GpuGzipCompressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Gzip
    }

    fn wrap_writer(self: Box<Self>, w: Box<dyn Write + Send>) -> Box<dyn Write + Send> {
        let lvl = self.level.clamp_to(0, 9) as u32;
        let identity = Arc::clone(&self.identity);
        let chunk_fn: ChunkFn = Arc::new(move |bytes: &[u8]| -> std::io::Result<Vec<u8>> {
            // Round-trip through the GPU. Identity for now; replaced by real
            // compression shaders in later tasks.
            let processed = identity.apply(bytes);
            let mut e = GzEncoder::new(Vec::with_capacity(bytes.len() / 2), Compression::new(lvl));
            e.write_all(&processed)?;
            e.finish()
        });
        Box::new(ParallelChunkedWriter::new(
            w,
            self.chunk_size,
            self.max_in_flight,
            chunk_fn,
        ))
    }
}
