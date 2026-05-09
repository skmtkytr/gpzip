//! CPU codec backend.
//!
//! Implements compression for Deflate / Zstd (formats gpzip writes) and
//! decompression for everything Ark opens (Deflate / Zstd / Lzma / Bzip2 /
//! Rar). Uses well-tested crates: flate2, zstd, xz2, bzip2, unrar.

mod backend;
mod compress;
mod decompress;

pub use backend::CpuBackend;
