//! Archive container abstraction (zip / tar / tar+compressor).
//!
//! Containers know about file metadata and structure; codecs know about
//! byte-level compression. Splitting them lets us reuse the same codec
//! backend across `.zip`, `.tar.gz`, `.tar.zst`, etc.

pub mod entry;
pub mod format;
pub mod list;
pub mod pack;
pub mod unpack;

pub use entry::Entry;
pub use format::{detect_format, ArchiveFormat};
pub use list::list_archive;
pub use pack::pack;
pub use unpack::unpack;
