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
