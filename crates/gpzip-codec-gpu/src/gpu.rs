//! Real GPU backend (wgpu). Skeleton only — `compressor` returns a CPU
//! fallback path until the WGSL shaders are wired in (task #10).

use std::sync::Arc;

use gpzip_core::{
    Algorithm, Capability, CodecBackend, Compressor, Decompressor, Error, Level, Result,
};

mod chunk;
mod context;

pub use context::GpuContext;

/// GPU codec. Holds an initialized wgpu device + queue. Cheap to clone
/// (Arc'd internally).
#[derive(Clone)]
pub struct GpuBackend {
    ctx: Arc<GpuContext>,
}

impl GpuBackend {
    pub const NAME: &'static str = "gpu";

    /// Probe for an adapter and initialize a device. Returns `Err` if no GPU
    /// is available so callers can fall back to CPU.
    pub fn try_init() -> Result<Self> {
        let ctx =
            GpuContext::try_init().map_err(|e| Error::Codec(format!("wgpu init failed: {e}")))?;
        Ok(Self { ctx: Arc::new(ctx) })
    }

    pub fn context(&self) -> &GpuContext {
        &self.ctx
    }
}

impl CodecBackend for GpuBackend {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn supports(&self, _algo: Algorithm) -> Capability {
        // No algorithms are wired up yet — actual GPU Deflate lands with the
        // WGSL shaders in task #10. Reporting `None` here makes the registry
        // fall through to CpuBackend so `--backend auto` still produces valid
        // output. Switch Deflate to `CompressOnly` once shaders are working.
        Capability::None
    }

    fn compressor(&self, algo: Algorithm, _level: Level) -> Result<Box<dyn Compressor>> {
        match algo {
            Algorithm::Deflate => Err(Error::Codec(
                "GPU Deflate compressor not yet implemented (task #10)".into(),
            )),
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
