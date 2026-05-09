use std::io::{Read, Write};

use crate::algorithm::{Algorithm, Capability, Level};
use crate::error::Result;

/// Streaming compressor. Wraps an output writer; bytes written to the wrapper
/// are compressed and forwarded. Compressed-stream finalization (footer,
/// flush) happens on drop.
pub trait Compressor: Send {
    fn algorithm(&self) -> Algorithm;
    fn wrap_writer(self: Box<Self>, w: Box<dyn Write + Send>) -> Box<dyn Write + Send>;
}

/// Streaming decompressor. Wraps an input reader; bytes read from the wrapper
/// are decompressed.
pub trait Decompressor: Send {
    fn algorithm(&self) -> Algorithm;
    fn wrap_reader(self: Box<Self>, r: Box<dyn Read + Send>) -> Box<dyn Read + Send>;
}

/// A codec backend (CPU / CUDA / Vulkan / ...). Stateless factory.
pub trait CodecBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn supports(&self, algo: Algorithm) -> Capability;
    fn compressor(&self, algo: Algorithm, level: Level) -> Result<Box<dyn Compressor>>;
    fn decompressor(&self, algo: Algorithm) -> Result<Box<dyn Decompressor>>;
}
