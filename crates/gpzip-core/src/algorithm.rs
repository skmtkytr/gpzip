use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Algorithm {
    /// Raw DEFLATE bitstream (used per-entry inside ZIP).
    Deflate,
    /// Gzip-wrapped DEFLATE (header + raw deflate + crc32/isize footer).
    /// Used by `.tar.gz` / `.gz`.
    Gzip,
    Zstd,
    Lzma,
    Bzip2,
    Rar,
}

impl fmt::Display for Algorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Deflate => "deflate",
            Self::Gzip => "gzip",
            Self::Zstd => "zstd",
            Self::Lzma => "lzma",
            Self::Bzip2 => "bzip2",
            Self::Rar => "rar",
        };
        f.write_str(s)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capability {
    None,
    DecompressOnly,
    CompressOnly,
    Both,
}

impl Capability {
    pub fn can_decompress(self) -> bool {
        matches!(self, Capability::DecompressOnly | Capability::Both)
    }
    pub fn can_compress(self) -> bool {
        matches!(self, Capability::CompressOnly | Capability::Both)
    }
}

/// Compression level. Backends interpret this as best they can.
/// 0 = fastest, 9 = best ratio, with 5 as a sensible default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Level(pub u8);

impl Default for Level {
    fn default() -> Self {
        Self(5)
    }
}

impl Level {
    pub fn clamp_to(self, min: i32, max: i32) -> i32 {
        let v = self.0 as i32;
        v.clamp(min, max)
    }
}
