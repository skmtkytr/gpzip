//! gpzip core: codec traits, archive I/O, progress events.
//!
//! UI- and GPU-agnostic. CLI/GUI frontends and CUDA/Vulkan/CPU backends all
//! depend on this crate.

pub mod algorithm;
pub mod archive;
pub mod codec;
pub mod error;
pub mod progress;
pub mod registry;

pub use algorithm::{Algorithm, Capability, Level};
pub use codec::{CodecBackend, Compressor, Decompressor};
pub use error::{Error, Result};
pub use progress::{ProgressEvent, ProgressSink};
pub use registry::BackendRegistry;
