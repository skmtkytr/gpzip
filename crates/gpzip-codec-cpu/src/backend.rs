use gpzip_core::{
    Algorithm, Capability, CodecBackend, Compressor, Decompressor, Error, Level, Result,
};

use crate::compress::{DeflateCompressor, GzipCompressor, ZstdCompressor};
use crate::decompress::{
    Bzip2Decompressor, DeflateDecompressor, GzipDecompressor, LzmaDecompressor, RarDecompressor,
    ZstdDecompressor,
};

#[derive(Default, Clone, Copy)]
pub struct CpuBackend;

impl CpuBackend {
    pub const NAME: &'static str = "cpu";
    pub fn new() -> Self {
        Self
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
            Algorithm::Deflate => Ok(Box::new(DeflateCompressor::new(level))),
            Algorithm::Gzip => Ok(Box::new(GzipCompressor::new(level))),
            Algorithm::Zstd => Ok(Box::new(ZstdCompressor::new(level))),
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
