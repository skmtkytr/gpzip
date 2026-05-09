use std::io::Read;

use bzip2::read::BzDecoder;
use flate2::read::{DeflateDecoder, MultiGzDecoder};
use gpzip_core::{Algorithm, Decompressor};
use xz2::read::XzDecoder;

pub struct DeflateDecompressor;
impl Decompressor for DeflateDecompressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Deflate
    }
    fn wrap_reader(self: Box<Self>, r: Box<dyn Read + Send>) -> Box<dyn Read + Send> {
        Box::new(DeflateDecoder::new(r))
    }
}

/// Gzip decoder. `MultiGzDecoder` handles concatenated gzip members, which
/// real-world `.tar.gz` files sometimes contain.
pub struct GzipDecompressor;
impl Decompressor for GzipDecompressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Gzip
    }
    fn wrap_reader(self: Box<Self>, r: Box<dyn Read + Send>) -> Box<dyn Read + Send> {
        Box::new(MultiGzDecoder::new(r))
    }
}

pub struct ZstdDecompressor;
impl Decompressor for ZstdDecompressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Zstd
    }
    fn wrap_reader(self: Box<Self>, r: Box<dyn Read + Send>) -> Box<dyn Read + Send> {
        Box::new(zstd::stream::read::Decoder::new(r).expect("zstd decoder init"))
    }
}

pub struct LzmaDecompressor;
impl Decompressor for LzmaDecompressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Lzma
    }
    fn wrap_reader(self: Box<Self>, r: Box<dyn Read + Send>) -> Box<dyn Read + Send> {
        Box::new(XzDecoder::new(r))
    }
}

pub struct Bzip2Decompressor;
impl Decompressor for Bzip2Decompressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Bzip2
    }
    fn wrap_reader(self: Box<Self>, r: Box<dyn Read + Send>) -> Box<dyn Read + Send> {
        Box::new(BzDecoder::new(r))
    }
}

/// RAR is decompressed via the `unrar` C library — it operates on whole
/// archives, not byte streams. The `wrap_reader` interface here is wrong for
/// it, so RAR support lives at the archive layer (see `gpzip-core::archive`).
/// This wrapper exists only so the codec lookup succeeds; it returns an error
/// on first read.
pub struct RarDecompressor;
impl Decompressor for RarDecompressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Rar
    }
    fn wrap_reader(self: Box<Self>, _r: Box<dyn Read + Send>) -> Box<dyn Read + Send> {
        Box::new(NotAStream)
    }
}

struct NotAStream;
impl Read for NotAStream {
    fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
        Err(std::io::Error::other(
            "RAR is not a streaming codec; use the archive-layer extractor",
        ))
    }
}
