//! RAR extraction. Compression isn't possible (closed format, UnRAR
//! license forbids derivative compressors), so this module only reads.
//!
//! Wraps the `unrar` crate, which links the official UnRAR C library.
//! Streaming over a pre-opened file handle isn't supported by the C API —
//! it always wants a path — so we just take a path here and let the
//! library do its thing.

use std::path::{Path, PathBuf};

use gpzip_core::archive::Entry;
use gpzip_core::{Error, ProgressEvent, ProgressSink, Result};
use unrar::Archive;

pub fn list_rar(archive_path: &Path) -> Result<Vec<Entry>> {
    let archive = Archive::new(archive_path)
        .open_for_listing()
        .map_err(|e| Error::InvalidArchive(format!("rar open: {e}")))?;
    let mut out = Vec::new();
    for entry in archive {
        let e = entry.map_err(|e| Error::InvalidArchive(format!("rar entry: {e}")))?;
        out.push(Entry {
            path: e.filename.clone(),
            size: e.unpacked_size,
            is_dir: e.is_directory(),
        });
    }
    Ok(out)
}

pub fn extract_rar(archive_path: &Path, dest: &Path, sink: &ProgressSink) -> Result<()> {
    std::fs::create_dir_all(dest)?;
    let dest_buf: PathBuf = dest.to_path_buf();
    sink.send(ProgressEvent::Started { total_bytes: 0 });

    let result = extract_inner(archive_path, &dest_buf, sink);
    match &result {
        Ok(_) => sink.send(ProgressEvent::Finished),
        Err(e) => sink.send(ProgressEvent::Error(e.to_string())),
    }
    result
}

fn extract_inner(archive_path: &Path, dest: &Path, sink: &ProgressSink) -> Result<()> {
    let mut archive = Archive::new(archive_path)
        .open_for_processing()
        .map_err(|e| Error::InvalidArchive(format!("rar open: {e}")))?;

    while let Some(header) = archive
        .read_header()
        .map_err(|e| Error::InvalidArchive(format!("rar header: {e}")))?
    {
        let entry = header.entry();
        let name = entry.filename.to_string_lossy().into_owned();
        let size = entry.unpacked_size;
        let is_file = entry.is_file();

        sink.send(ProgressEvent::FileStarted {
            path: name.clone(),
            size,
        });

        archive = if is_file {
            header
                .extract_with_base(dest)
                .map_err(|e| Error::InvalidArchive(format!("rar extract `{name}`: {e}")))?
        } else {
            header
                .skip()
                .map_err(|e| Error::InvalidArchive(format!("rar skip `{name}`: {e}")))?
        };

        sink.send(ProgressEvent::FileFinished { path: name });
    }
    Ok(())
}
