use thiserror::Error;

use crate::algorithm::Algorithm;

#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("backend `{backend}` does not support algorithm {algo:?}")]
    UnsupportedAlgorithm {
        backend: &'static str,
        algo: Algorithm,
    },

    #[error("backend `{backend}` cannot compress with {algo:?} (decompress-only)")]
    CompressionUnsupported {
        backend: &'static str,
        algo: Algorithm,
    },

    #[error("backend `{backend}` cannot decompress {algo:?} (compress-only)")]
    DecompressionUnsupported {
        backend: &'static str,
        algo: Algorithm,
    },

    #[error("no backend available for {algo:?}")]
    NoBackend { algo: Algorithm },

    #[error("invalid archive: {0}")]
    InvalidArchive(String),

    #[error("codec error: {0}")]
    Codec(String),

    #[error("zip error: {0}")]
    Zip(#[from] zip::result::ZipError),
}

pub type Result<T> = std::result::Result<T, Error>;
