use gpzip_core::{
    Algorithm, Capability, CodecBackend, Compressor, Decompressor, Error, Level, Result,
};

use crate::compress::{DeflateCompressor, GzipCompressor, ZstdCompressor};
use crate::decompress::{
    Bzip2Decompressor, DeflateDecompressor, GzipDecompressor, LzmaDecompressor, RarDecompressor,
    ZstdDecompressor,
};

/// Default chunk size for parallel compression. Matches cozip's CMP starting
/// point of 2 MiB; large enough that ratio loss vs serial is small (~1-2%),
/// small enough that even small files get some chunking.
pub const DEFAULT_CHUNK_SIZE: usize = 2 * 1024 * 1024;

#[derive(Clone, Copy)]
pub struct CpuBackend {
    chunk_size: usize,
    max_in_flight: usize,
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuBackend {
    pub const NAME: &'static str = "cpu";

    pub fn new() -> Self {
        Self {
            chunk_size: DEFAULT_CHUNK_SIZE,
            max_in_flight: num_cpus::get().max(1),
        }
    }

    /// Configure parallelism. `max_in_flight = 1` is effectively serial.
    /// `chunk_size` of 0 is rejected (would deadlock the writer).
    pub fn with_config(chunk_size: usize, max_in_flight: usize) -> Self {
        assert!(chunk_size > 0, "chunk_size must be > 0");
        assert!(max_in_flight > 0, "max_in_flight must be > 0");
        Self {
            chunk_size,
            max_in_flight,
        }
    }

    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    pub fn max_in_flight(&self) -> usize {
        self.max_in_flight
    }
}

impl CodecBackend for CpuBackend {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn supports(&self, algo: Algorithm) -> Capability {
        match algo {
            Algorithm::Deflate => Capability::Both,
            Algorithm::Gzip => Capability::Both,
            Algorithm::Zstd => Capability::Both,
            Algorithm::Lzma => Capability::DecompressOnly,
            Algorithm::Bzip2 => Capability::DecompressOnly,
            Algorithm::Rar => Capability::DecompressOnly,
        }
    }

    fn compressor(&self, algo: Algorithm, level: Level) -> Result<Box<dyn Compressor>> {
        match algo {
            // Raw deflate has no member framing, so per-chunk concat would
            // not be readable by zip-rs. Keep it serial.
            Algorithm::Deflate => Ok(Box::new(DeflateCompressor::new(level))),
            // Gzip and zstd both support concatenated members/frames as part
            // of their formats — perfect for chunk-parallel output.
            Algorithm::Gzip => Ok(Box::new(GzipCompressor::new(
                level,
                self.chunk_size,
                self.max_in_flight,
            ))),
            Algorithm::Zstd => Ok(Box::new(ZstdCompressor::new(
                level,
                self.chunk_size,
                self.max_in_flight,
            ))),
            Algorithm::Lzma | Algorithm::Bzip2 | Algorithm::Rar => {
                Err(Error::CompressionUnsupported {
                    backend: Self::NAME,
                    algo,
                })
            }
        }
    }

    fn decompressor(&self, algo: Algorithm) -> Result<Box<dyn Decompressor>> {
        match algo {
            Algorithm::Deflate => Ok(Box::new(DeflateDecompressor)),
            Algorithm::Gzip => Ok(Box::new(GzipDecompressor)),
            Algorithm::Zstd => Ok(Box::new(ZstdDecompressor)),
            Algorithm::Lzma => Ok(Box::new(LzmaDecompressor)),
            Algorithm::Bzip2 => Ok(Box::new(Bzip2Decompressor)),
            Algorithm::Rar => Ok(Box::new(RarDecompressor)),
        }
    }
}
