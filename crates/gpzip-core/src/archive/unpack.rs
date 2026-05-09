use std::fs::{self, File};
use std::io::{self, Cursor};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use rayon::prelude::*;

use crate::archive::format::{detect_format, ArchiveFormat};
use crate::archive::list::open_tar_reader;
use crate::error::{Error, Result};
use crate::progress::{ProgressEvent, ProgressSink};
use crate::registry::BackendRegistry;

pub fn unpack(
    archive_path: &Path,
    dest: &Path,
    registry: &BackendRegistry,
    sink: ProgressSink,
) -> Result<()> {
    let format = detect_format(archive_path).ok_or_else(|| {
        Error::InvalidArchive(format!("unknown format: {}", archive_path.display()))
    })?;

    fs::create_dir_all(dest)?;
    sink.send(ProgressEvent::Started { total_bytes: 0 });

    let result = match format {
        ArchiveFormat::Zip => unpack_zip(archive_path, dest, &sink),
        ArchiveFormat::Tar
        | ArchiveFormat::TarGz
        | ArchiveFormat::TarZst
        | ArchiveFormat::TarXz
        | ArchiveFormat::TarBz2 => unpack_tar(archive_path, format, dest, registry, &sink),
        ArchiveFormat::Rar | ArchiveFormat::SevenZ => Err(Error::InvalidArchive(format!(
            "{format:?} extraction must be invoked via gpzip-codec-cpu"
        ))),
    };

    match &result {
        Ok(_) => sink.send(ProgressEvent::Finished),
        Err(e) => sink.send(ProgressEvent::Error(e.to_string())),
    }
    result
}

fn unpack_zip(archive: &Path, dest: &Path, sink: &ProgressSink) -> Result<()> {
    // ZIP entries are independent — each has its own deflate stream and
    // local file header. Read the file once, share via Arc, and let rayon
    // spread entry decompression + write across cores. Each worker rebuilds
    // a ZipArchive over the shared bytes; the central directory parse is
    // cheap (microseconds for thousands of entries) and isn't worth caching.

    // Arc<[u8]> impls AsRef<[u8]> directly, so Cursor<Arc<[u8]>> is Read+Seek
    // and can be shared across threads without copies.
    let bytes: Arc<[u8]> = fs::read(archive)?.into();
    let n = zip::ZipArchive::new(Cursor::new(Arc::clone(&bytes)))?.len();
    let dest = dest.to_path_buf();

    (0..n).into_par_iter().try_for_each_init(
        || zip::ZipArchive::new(Cursor::new(Arc::clone(&bytes))).expect("zip already validated"),
        |zip, i| -> Result<()> {
            let mut entry = zip.by_index(i)?;
            let raw = entry.mangled_name();
            let target = safe_join(&dest, &raw)?;
            sink.send(ProgressEvent::FileStarted {
                path: target.display().to_string(),
                size: entry.size(),
            });

            if entry.is_dir() {
                fs::create_dir_all(&target)?;
            } else {
                if let Some(p) = target.parent() {
                    fs::create_dir_all(p)?;
                }
                let mut out = File::create(&target)?;
                io::copy(&mut entry, &mut out)?;
            }
            sink.send(ProgressEvent::FileFinished {
                path: target.display().to_string(),
            });
            Ok(())
        },
    )?;
    Ok(())
}

fn unpack_tar(
    archive: &Path,
    format: ArchiveFormat,
    dest: &Path,
    registry: &BackendRegistry,
    sink: &ProgressSink,
) -> Result<()> {
    let reader = open_tar_reader(archive, format, registry)?;
    let mut tar = tar::Archive::new(reader);
    // tar's `unpack` does the safe path checks itself (relative paths,
    // strip absolute prefixes, no `..` escape).
    for entry in tar.entries()? {
        let mut e = entry?;
        let path = e.path()?.into_owned();
        sink.send(ProgressEvent::FileStarted {
            path: path.display().to_string(),
            size: e.size(),
        });
        e.unpack_in(dest)?;
        sink.send(ProgressEvent::FileFinished {
            path: path.display().to_string(),
        });
    }
    Ok(())
}

/// Reject absolute paths and any `..` segments. Returns `dest` joined with
/// the entry's path. Defense-in-depth for zip slip.
fn safe_join(dest: &Path, entry: &Path) -> Result<PathBuf> {
    let mut out = dest.to_path_buf();
    for c in entry.components() {
        match c {
            Component::Normal(n) => out.push(n),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(Error::InvalidArchive(format!(
                    "unsafe entry path: {}",
                    entry.display()
                )));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn safe_join_rejects_traversal() {
        assert!(safe_join(Path::new("/tmp"), &PathBuf::from("../etc/passwd")).is_err());
        assert!(safe_join(Path::new("/tmp"), &PathBuf::from("/etc/passwd")).is_err());
    }

    #[test]
    fn safe_join_accepts_normal_paths() {
        let p = safe_join(Path::new("/tmp"), &PathBuf::from("a/b/c.txt")).unwrap();
        assert_eq!(p, PathBuf::from("/tmp/a/b/c.txt"));
    }
}
