//! 7z extraction. Compression is out of scope for v1 — common 7z archives
//! use LZMA2, which is dominated by sequential decode dependencies and isn't
//! a good fit for the chunk-parallel pipeline.
//!
//! Wraps `sevenz-rust2` (pure-Rust 7z reader/writer). Encrypted archives are
//! rejected up front: passwords are a CLI flag we haven't added yet.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use gpzip_core::archive::Entry;
use gpzip_core::{Error, ProgressEvent, ProgressSink, Result};
use sevenz_rust2::ArchiveReader;

const NO_PASSWORD: &str = "";

pub fn list_sevenz(archive_path: &Path) -> Result<Vec<Entry>> {
    let reader = ArchiveReader::open(archive_path, NO_PASSWORD.into())
        .map_err(|e| Error::InvalidArchive(format!("7z open: {e}")))?;
    let mut out = Vec::new();
    for f in &reader.archive().files {
        out.push(Entry {
            path: PathBuf::from(f.name()),
            size: f.size(),
            is_dir: f.is_directory(),
        });
    }
    Ok(out)
}

pub fn extract_sevenz(archive_path: &Path, dest: &Path, sink: &ProgressSink) -> Result<()> {
    std::fs::create_dir_all(dest)?;
    sink.send(ProgressEvent::Started { total_bytes: 0 });
    let result = extract_inner(archive_path, dest, sink);
    match &result {
        Ok(_) => sink.send(ProgressEvent::Finished),
        Err(e) => sink.send(ProgressEvent::Error(e.to_string())),
    }
    result
}

fn extract_inner(archive_path: &Path, dest: &Path, sink: &ProgressSink) -> Result<()> {
    let mut sz = ArchiveReader::open(archive_path, NO_PASSWORD.into())
        .map_err(|e| Error::InvalidArchive(format!("7z open: {e}")))?;

    let dest = dest.to_path_buf();
    sz.for_each_entries(|entry, reader| {
        let name = entry.name().to_string();
        let size = entry.size();
        let target = safe_join(&dest, Path::new(&name)).map_err(std::io::Error::other)?;

        if entry.is_directory() {
            std::fs::create_dir_all(&target)?;
            return Ok(true);
        }

        sink.send(ProgressEvent::FileStarted {
            path: name.clone(),
            size,
        });
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = File::create(&target)?;
        let mut buf = [0u8; 64 * 1024];
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n])?;
        }
        sink.send(ProgressEvent::FileFinished { path: name });
        Ok(true)
    })
    .map_err(|e| Error::InvalidArchive(format!("7z extract: {e}")))?;
    Ok(())
}

/// Reject absolute paths and `..` segments — defense-in-depth against
/// archive-traversal entries.
fn safe_join(dest: &Path, entry: &Path) -> std::result::Result<PathBuf, String> {
    use std::path::Component;
    let mut out = dest.to_path_buf();
    for c in entry.components() {
        match c {
            Component::Normal(n) => out.push(n),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(format!("unsafe entry path: {}", entry.display()));
            }
        }
    }
    Ok(out)
}
