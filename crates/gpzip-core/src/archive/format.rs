use std::path::Path;

use crate::algorithm::Algorithm;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveFormat {
    Zip,
    Tar,
    TarGz,
    TarZst,
    TarXz,
    TarBz2,
    Rar,
    SevenZ,
}

impl ArchiveFormat {
    /// Algorithm used by this container's payload, if any.
    pub fn payload_algorithm(self) -> Option<Algorithm> {
        match self {
            // ZIP wraps each entry in raw DEFLATE inside its own header.
            Self::Zip => Some(Algorithm::Deflate),
            // tar.gz / .gz are gzip-wrapped DEFLATE (header + footer).
            Self::TarGz => Some(Algorithm::Gzip),
            Self::TarZst => Some(Algorithm::Zstd),
            Self::TarXz | Self::SevenZ => Some(Algorithm::Lzma),
            Self::TarBz2 => Some(Algorithm::Bzip2),
            Self::Rar => Some(Algorithm::Rar),
            Self::Tar => None,
        }
    }

    /// True if gpzip can produce this format (compression supported).
    pub fn is_writable(self) -> bool {
        matches!(self, Self::Zip | Self::Tar | Self::TarGz | Self::TarZst)
    }
}

/// Best-effort format detection from a file path / name.
pub fn detect_format(path: &Path) -> Option<ArchiveFormat> {
    let name = path.file_name()?.to_str()?.to_ascii_lowercase();

    // Check compound extensions first.
    if name.ends_with(".tar.gz") || name.ends_with(".tgz") {
        return Some(ArchiveFormat::TarGz);
    }
    if name.ends_with(".tar.zst") || name.ends_with(".tzst") {
        return Some(ArchiveFormat::TarZst);
    }
    if name.ends_with(".tar.xz") || name.ends_with(".txz") {
        return Some(ArchiveFormat::TarXz);
    }
    if name.ends_with(".tar.bz2") || name.ends_with(".tbz2") || name.ends_with(".tbz") {
        return Some(ArchiveFormat::TarBz2);
    }
    if name.ends_with(".tar") {
        return Some(ArchiveFormat::Tar);
    }
    if name.ends_with(".zip") {
        return Some(ArchiveFormat::Zip);
    }
    if name.ends_with(".rar") {
        return Some(ArchiveFormat::Rar);
    }
    if name.ends_with(".7z") {
        return Some(ArchiveFormat::SevenZ);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn detect(s: &str) -> Option<ArchiveFormat> {
        detect_format(&PathBuf::from(s))
    }

    #[test]
    fn detects_zip() {
        assert_eq!(detect("foo.zip"), Some(ArchiveFormat::Zip));
        assert_eq!(detect("FOO.ZIP"), Some(ArchiveFormat::Zip));
    }

    #[test]
    fn detects_compound_tar_extensions() {
        assert_eq!(detect("a.tar.gz"), Some(ArchiveFormat::TarGz));
        assert_eq!(detect("a.tgz"), Some(ArchiveFormat::TarGz));
        assert_eq!(detect("a.tar.zst"), Some(ArchiveFormat::TarZst));
        assert_eq!(detect("a.tar.xz"), Some(ArchiveFormat::TarXz));
        assert_eq!(detect("a.tar.bz2"), Some(ArchiveFormat::TarBz2));
        assert_eq!(detect("a.tar"), Some(ArchiveFormat::Tar));
    }

    #[test]
    fn detects_rar_and_7z() {
        assert_eq!(detect("a.rar"), Some(ArchiveFormat::Rar));
        assert_eq!(detect("a.7z"), Some(ArchiveFormat::SevenZ));
    }

    #[test]
    fn unknown_returns_none() {
        assert_eq!(detect("a.unknown"), None);
        assert_eq!(detect("noext"), None);
    }

    #[test]
    fn writable_set_is_minimal() {
        assert!(ArchiveFormat::Zip.is_writable());
        assert!(ArchiveFormat::TarGz.is_writable());
        assert!(ArchiveFormat::TarZst.is_writable());
        assert!(!ArchiveFormat::TarXz.is_writable());
        assert!(!ArchiveFormat::Rar.is_writable());
        assert!(!ArchiveFormat::SevenZ.is_writable());
    }

    #[test]
    fn payload_algorithm_mapping() {
        assert_eq!(
            ArchiveFormat::Zip.payload_algorithm(),
            Some(Algorithm::Deflate)
        );
        assert_eq!(
            ArchiveFormat::TarGz.payload_algorithm(),
            Some(Algorithm::Gzip)
        );
        assert_eq!(
            ArchiveFormat::TarZst.payload_algorithm(),
            Some(Algorithm::Zstd)
        );
        assert_eq!(ArchiveFormat::Tar.payload_algorithm(), None);
    }
}
