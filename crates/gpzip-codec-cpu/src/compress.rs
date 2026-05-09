use std::io::Write;

use flate2::{
    write::{DeflateEncoder, GzEncoder},
    Compression,
};
use gpzip_core::{Algorithm, Compressor, Level};

pub struct DeflateCompressor {
    level: Level,
}
impl DeflateCompressor {
    pub fn new(level: Level) -> Self {
        Self { level }
    }
}
impl Compressor for DeflateCompressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Deflate
    }
    fn wrap_writer(self: Box<Self>, w: Box<dyn Write + Send>) -> Box<dyn Write + Send> {
        let lvl = self.level.clamp_to(0, 9) as u32;
        Box::new(DeflateEncoder::new(w, Compression::new(lvl)))
    }
}

/// Gzip-wrapped DEFLATE. Used for `.tar.gz` / `.gz`.
pub struct GzipCompressor {
    level: Level,
}
impl GzipCompressor {
    pub fn new(level: Level) -> Self {
        Self { level }
    }
}
impl Compressor for GzipCompressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Gzip
    }
    fn wrap_writer(self: Box<Self>, w: Box<dyn Write + Send>) -> Box<dyn Write + Send> {
        let lvl = self.level.clamp_to(0, 9) as u32;
        Box::new(GzEncoder::new(w, Compression::new(lvl)))
    }
}

pub struct ZstdCompressor {
    level: Level,
}
impl ZstdCompressor {
    pub fn new(level: Level) -> Self {
        Self { level }
    }
}
impl Compressor for ZstdCompressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Zstd
    }
    fn wrap_writer(self: Box<Self>, w: Box<dyn Write + Send>) -> Box<dyn Write + Send> {
        // zstd levels: 1..=22, 3 default. Map our 0..=9 to a useful sub-range.
        // 0 -> 1, 5 -> 3 (default), 9 -> 19 (high effort but not max-22).
        let mapped = match self.level.0 {
            0 => 1,
            1..=4 => 2,
            5 => 3,
            6 => 6,
            7 => 12,
            8 => 17,
            _ => 19,
        };
        // AutoFinishEncoder finalizes the frame on drop.
        Box::new(
            zstd::stream::write::Encoder::new(w, mapped)
                .expect("zstd encoder init")
                .auto_finish(),
        )
    }
}
