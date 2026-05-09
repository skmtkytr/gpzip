use std::sync::Arc;

use crate::algorithm::Algorithm;
use crate::codec::CodecBackend;
use crate::error::{Error, Result};

/// Holds an ordered list of backends. Earlier entries are preferred. Use
/// `with_priority` to put GPU before CPU; falls back automatically when a
/// backend doesn't support an algorithm.
#[derive(Clone, Default)]
pub struct BackendRegistry {
    backends: Vec<Arc<dyn CodecBackend>>,
}

impl BackendRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, backend: Arc<dyn CodecBackend>) {
        self.backends.push(backend);
    }

    pub fn names(&self) -> Vec<&'static str> {
        self.backends.iter().map(|b| b.name()).collect()
    }

    /// Find first backend that can compress `algo`.
    pub fn pick_compressor(&self, algo: Algorithm) -> Result<Arc<dyn CodecBackend>> {
        self.backends
            .iter()
            .find(|b| b.supports(algo).can_compress())
            .cloned()
            .ok_or(Error::NoBackend { algo })
    }

    /// Find first backend that can decompress `algo`.
    pub fn pick_decompressor(&self, algo: Algorithm) -> Result<Arc<dyn CodecBackend>> {
        self.backends
            .iter()
            .find(|b| b.supports(algo).can_decompress())
            .cloned()
            .ok_or(Error::NoBackend { algo })
    }

    /// Find a specific backend by name (for `--backend cpu`).
    pub fn by_name(&self, name: &str) -> Option<Arc<dyn CodecBackend>> {
        self.backends.iter().find(|b| b.name() == name).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::{Capability, Level};
    use crate::codec::{Compressor, Decompressor};

    struct MockBackend {
        name: &'static str,
        cap: Capability,
        supported: Algorithm,
    }
    impl CodecBackend for MockBackend {
        fn name(&self) -> &'static str {
            self.name
        }
        fn supports(&self, algo: Algorithm) -> Capability {
            if algo == self.supported {
                self.cap
            } else {
                Capability::None
            }
        }
        fn compressor(&self, _: Algorithm, _: Level) -> Result<Box<dyn Compressor>> {
            unimplemented!()
        }
        fn decompressor(&self, _: Algorithm) -> Result<Box<dyn Decompressor>> {
            unimplemented!()
        }
    }

    #[test]
    fn picks_first_capable_backend() {
        let mut reg = BackendRegistry::new();
        reg.push(Arc::new(MockBackend {
            name: "gpu",
            cap: Capability::Both,
            supported: Algorithm::Deflate,
        }));
        reg.push(Arc::new(MockBackend {
            name: "cpu",
            cap: Capability::Both,
            supported: Algorithm::Deflate,
        }));
        let chosen = reg.pick_compressor(Algorithm::Deflate).unwrap();
        assert_eq!(chosen.name(), "gpu");
    }

    #[test]
    fn falls_back_when_first_does_not_support() {
        let mut reg = BackendRegistry::new();
        reg.push(Arc::new(MockBackend {
            name: "gpu",
            cap: Capability::None,
            supported: Algorithm::Deflate,
        }));
        reg.push(Arc::new(MockBackend {
            name: "cpu",
            cap: Capability::Both,
            supported: Algorithm::Deflate,
        }));
        let chosen = reg.pick_compressor(Algorithm::Deflate).unwrap();
        assert_eq!(chosen.name(), "cpu");
    }

    #[test]
    fn decompress_only_backend_cannot_compress() {
        let mut reg = BackendRegistry::new();
        reg.push(Arc::new(MockBackend {
            name: "cpu",
            cap: Capability::DecompressOnly,
            supported: Algorithm::Rar,
        }));
        assert!(reg.pick_compressor(Algorithm::Rar).is_err());
        assert!(reg.pick_decompressor(Algorithm::Rar).is_ok());
    }

    #[test]
    fn no_backend_returns_error() {
        let reg = BackendRegistry::new();
        // pick_compressor returns Result<Arc<dyn CodecBackend>, Error>; Arc is
        // not Debug so we match on the result rather than unwrap_err().
        match reg.pick_compressor(Algorithm::Zstd) {
            Err(Error::NoBackend {
                algo: Algorithm::Zstd,
            }) => {}
            other => panic!("expected NoBackend(Zstd), got {:?}", other.is_err()),
        }
    }
}
