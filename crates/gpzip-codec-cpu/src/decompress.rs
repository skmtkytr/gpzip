use std::io::{Cursor, Read};

use bzip2::read::BzDecoder;
use flate2::read::{DeflateDecoder, MultiGzDecoder};
use gpzip_core::{Algorithm, Decompressor};
use xz2::read::XzDecoder;

use crate::parallel_decompress;

pub struct DeflateDecompressor;
impl Decompressor for DeflateDecompressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Deflate
    }
    fn wrap_reader(self: Box<Self>, r: Box<dyn Read + Send>) -> Box<dyn Read + Send> {
        Box::new(DeflateDecoder::new(r))
    }
}

/// Gzip decoder. For gpzip-written archives (sequence of fixed-header
/// gzip members), slurps the input and decodes members in parallel via
/// `parallel_decompress`. For other inputs (system gzip with FNAME etc.)
/// the boundary scan finds nothing and falls back to serial
/// `MultiGzDecoder` over the buffered bytes — same correctness as before,
/// just at the cost of buffering the whole file in RAM. Worth it because
/// the typical gpzip user is extracting an archive they (or another
/// gpzip) created.
pub struct GzipDecompressor;
impl Decompressor for GzipDecompressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Gzip
    }
    fn wrap_reader(mut self: Box<Self>, mut r: Box<dyn Read + Send>) -> Box<dyn Read + Send> {
        let _ = &mut self;
        let mut compressed = Vec::new();
        if let Err(e) = r.read_to_end(&mut compressed) {
            // Surface the read error from the first .read() on the returned reader.
            return Box::new(ErrReader(Some(e)));
        }
        match parallel_decompress::parallel_decompress(&compressed) {
            Ok(decompressed) => Box::new(Cursor::new(decompressed)),
            Err(_) => Box::new(MultiGzDecoder::new(Cursor::new(compressed))),
        }
    }
}

/// Read-side adapter that surfaces a stored io::Error on first read.
/// Used when wrap_reader can't even slurp the input — without this we'd
/// have to panic or silently return EOF, both worse.
struct ErrReader(Option<std::io::Error>);
impl Read for ErrReader {
    fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
        match self.0.take() {
            Some(e) => Err(e),
            None => Ok(0),
        }
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
