use gpzip_codec_cpu::ChunkFn;
use gpzip_core::{
    Algorithm, Capability, CodecBackend, Compressor, Decompressor, Error, Level, Result,
};

/// Compiled in when the `enabled` feature is off. Reports no capabilities;
/// the registry falls through to the CPU backend.
#[derive(Default, Clone, Copy)]
pub struct StubBackend;

impl StubBackend {
    pub const NAME: &'static str = "gpu";
    pub fn try_init() -> Result<Self> {
        Ok(Self)
    }

    /// Stub of the real `GpuBackend::gzip_chunk_fn` — same signature so
    /// the CLI's hybrid path type-checks regardless of feature flag.
    /// Never called at runtime in stub mode (`StubLazyBackend::try_get`
    /// always returns `None`, so the closure that would invoke this is
    /// never produced).
    pub fn gzip_chunk_fn(&self) -> ChunkFn {
        std::sync::Arc::new(|_bytes: &[u8]| {
            Err(std::io::Error::other(
                "GPU codec stubbed out at compile time — rebuild with --features gpu",
            ))
        })
    }
}

/// Lazy stub: same shape as the real `LazyGpuBackend` so callers (the CLI's
/// hybrid path) compile against the same API regardless of feature flag.
/// `try_get` always returns `None` here.
#[derive(Default)]
pub struct StubLazyBackend;

impl StubLazyBackend {
    pub fn new() -> Self {
        Self
    }
    pub fn try_get(&self) -> Option<&std::sync::Arc<StubBackend>> {
        None
    }
}

impl CodecBackend for StubBackend {
    fn name(&self) -> &'static str {
        Self::NAME
    }
    fn supports(&self, _algo: Algorithm) -> Capability {
        Capability::None
    }
    fn compressor(&self, algo: Algorithm, _: Level) -> Result<Box<dyn Compressor>> {
        Err(Error::UnsupportedAlgorithm {
            backend: Self::NAME,
            algo,
        })
    }
    fn decompressor(&self, algo: Algorithm) -> Result<Box<dyn Decompressor>> {
        Err(Error::UnsupportedAlgorithm {
            backend: Self::NAME,
            algo,
        })
    }
}

impl CodecBackend for StubLazyBackend {
    fn name(&self) -> &'static str {
        StubBackend::NAME
    }
    fn supports(&self, _algo: Algorithm) -> Capability {
        Capability::None
    }
    fn compressor(&self, algo: Algorithm, _: Level) -> Result<Box<dyn Compressor>> {
        Err(Error::UnsupportedAlgorithm {
            backend: StubBackend::NAME,
            algo,
        })
    }
    fn decompressor(&self, algo: Algorithm) -> Result<Box<dyn Decompressor>> {
        Err(Error::UnsupportedAlgorithm {
            backend: StubBackend::NAME,
            algo,
        })
    }
}
