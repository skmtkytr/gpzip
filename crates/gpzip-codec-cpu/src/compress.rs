use std::io::Write;
use std::sync::Arc;

use flate2::{
    write::{DeflateEncoder, GzEncoder},
    Compression,
};
use gpzip_core::{Algorithm, Compressor, Level};

use crate::parallel::{ChunkFn, ParallelChunkedWriter};

/// Raw DEFLATE (no framing). Used inside per-entry ZIP compression.
/// Serial; ZIP doesn't allow chunking within a single entry.
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

/// Gzip-wrapped DEFLATE. Used for `.tar.gz` / `.gz`. Chunk-parallel: each
/// chunk becomes an independent gzip member; standard gunzip / MultiGzDecoder
/// concatenates them transparently (RFC 1952 §2.2).
pub struct GzipCompressor {
    level: Level,
    chunk_size: usize,
    max_in_flight: usize,
}
impl GzipCompressor {
    pub fn new(level: Level, chunk_size: usize, max_in_flight: usize) -> Self {
        Self {
            level,
            chunk_size,
            max_in_flight,
        }
    }
}
impl GzipCompressor {
    /// Per-chunk gzip-member encoder. Pulled out of `wrap_writer` so the
    /// hybrid CPU+GPU writer in the CLI can compose it with a GPU chunk_fn.
    pub fn chunk_fn(level: Level) -> ChunkFn {
        let lvl = level.clamp_to(0, 9) as u32;
        Arc::new(move |bytes: &[u8]| {
            let mut e = GzEncoder::new(Vec::with_capacity(bytes.len() / 2), Compression::new(lvl));
            e.write_all(bytes)?;
            e.finish()
        })
    }
}

impl Compressor for GzipCompressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Gzip
    }
    fn wrap_writer(self: Box<Self>, w: Box<dyn Write + Send>) -> Box<dyn Write + Send> {
        Box::new(ParallelChunkedWriter::new(
            w,
            self.chunk_size,
            self.max_in_flight,
            GzipCompressor::chunk_fn(self.level),
        ))
    }
}

/// Zstd. Used for `.tar.zst`. Chunk-parallel via concatenated zstd frames
/// (zstd format permits multiple frames in one stream).
pub struct ZstdCompressor {
    level: Level,
    chunk_size: usize,
    max_in_flight: usize,
}
impl ZstdCompressor {
    pub fn new(level: Level, chunk_size: usize, max_in_flight: usize) -> Self {
        Self {
            level,
            chunk_size,
            max_in_flight,
        }
    }
}
impl ZstdCompressor {
    /// Per-chunk zstd-frame encoder. Pulled out for hybrid composition.
    pub fn chunk_fn(level: Level) -> ChunkFn {
        // Map our 0..=9 onto zstd's 1..=22 with sensible defaults.
        let mapped: i32 = match level.0 {
            0 => 1,
            1..=4 => 2,
            5 => 3,
            6 => 6,
            7 => 12,
            8 => 17,
            _ => 19,
        };
        Arc::new(move |bytes: &[u8]| {
            let mut e =
                zstd::stream::write::Encoder::new(Vec::with_capacity(bytes.len() / 2), mapped)?;
            e.write_all(bytes)?;
            e.finish()
        })
    }
}

impl Compressor for ZstdCompressor {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Zstd
    }
    fn wrap_writer(self: Box<Self>, w: Box<dyn Write + Send>) -> Box<dyn Write + Send> {
        Box::new(ParallelChunkedWriter::new(
            w,
            self.chunk_size,
            self.max_in_flight,
            ZstdCompressor::chunk_fn(self.level),
        ))
    }
}
