use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::archive::entry::Entry;
use crate::archive::format::{detect_format, ArchiveFormat};
use crate::error::{Error, Result};
use crate::registry::BackendRegistry;

/// Enumerate entries in an archive without extracting.
pub fn list_archive(path: &Path, registry: &BackendRegistry) -> Result<Vec<Entry>> {
    let format = detect_format(path)
        .ok_or_else(|| Error::InvalidArchive(format!("unknown format: {}", path.display())))?;

    match format {
        ArchiveFormat::Zip => list_zip(path),
        ArchiveFormat::Tar
        | ArchiveFormat::TarGz
        | ArchiveFormat::TarZst
        | ArchiveFormat::TarXz
        | ArchiveFormat::TarBz2 => list_tar(path, format, registry),
        ArchiveFormat::Rar | ArchiveFormat::SevenZ => Err(Error::InvalidArchive(format!(
            "{format:?} listing not yet wired (use the codec-cpu archive extractor)"
        ))),
    }
}

fn list_zip(path: &Path) -> Result<Vec<Entry>> {
    let f = BufReader::new(File::open(path)?);
    let mut zip = zip::ZipArchive::new(f)?;
    let mut out = Vec::with_capacity(zip.len());
    for i in 0..zip.len() {
        let e = zip.by_index(i)?;
        out.push(Entry {
            path: e.mangled_name(),
            size: e.size(),
            is_dir: e.is_dir(),
        });
    }
    Ok(out)
}

fn list_tar(path: &Path, format: ArchiveFormat, registry: &BackendRegistry) -> Result<Vec<Entry>> {
    let reader = open_tar_reader(path, format, registry)?;
    let mut tar = tar::Archive::new(reader);
    let mut out = Vec::new();
    for entry in tar.entries()? {
        let e = entry?;
        let header = e.header();
        out.push(Entry {
            path: e.path()?.into_owned(),
            size: header.size().unwrap_or(0),
            is_dir: header.entry_type().is_dir(),
        });
    }
    Ok(out)
}

/// Open a tar(.gz|.zst|.xz|.bz2|...) for reading, dispatching the inner
/// decompressor through the registry. `Tar` returns the raw file.
pub(crate) fn open_tar_reader(
    path: &Path,
    format: ArchiveFormat,
    registry: &BackendRegistry,
) -> Result<Box<dyn Read + Send>> {
    let file: Box<dyn Read + Send> = Box::new(BufReader::new(File::open(path)?));
    let Some(algo) = format.payload_algorithm() else {
        return Ok(file);
    };
    let backend = registry.pick_decompressor(algo)?;
    let decoder = backend.decompressor(algo)?;
    Ok(decoder.wrap_reader(file))
}
