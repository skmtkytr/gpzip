use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::algorithm::Level;
use crate::archive::format::{detect_format, ArchiveFormat};
use crate::error::{Error, Result};
use crate::progress::{ProgressEvent, ProgressSink};
use crate::registry::BackendRegistry;

pub fn pack(
    archive_path: &Path,
    inputs: &[PathBuf],
    level: Level,
    registry: &BackendRegistry,
    sink: ProgressSink,
) -> Result<()> {
    let format = detect_format(archive_path).ok_or_else(|| {
        Error::InvalidArchive(format!("unknown format: {}", archive_path.display()))
    })?;
    if !format.is_writable() {
        return Err(Error::InvalidArchive(format!(
            "{format:?} is not writable; gpzip can only produce zip / tar / tar.gz / tar.zst"
        )));
    }

    sink.send(ProgressEvent::Started { total_bytes: 0 });

    let result = match format {
        ArchiveFormat::Zip => pack_zip(archive_path, inputs, level, &sink),
        ArchiveFormat::Tar | ArchiveFormat::TarGz | ArchiveFormat::TarZst => pack_tar(
            archive_path,
            inputs,
            format.payload_algorithm(),
            level,
            registry,
            &sink,
        ),
        _ => unreachable!("guarded by is_writable above"),
    };

    match &result {
        Ok(_) => sink.send(ProgressEvent::Finished),
        Err(e) => sink.send(ProgressEvent::Error(e.to_string())),
    }
    result
}

fn pack_zip(archive: &Path, inputs: &[PathBuf], level: Level, sink: &ProgressSink) -> Result<()> {
    let file = BufWriter::new(File::create(archive)?);
    let mut zw = zip::ZipWriter::new(file);
    let opts: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated)
        .compression_level(Some(level.clamp_to(0, 9) as i64));

    for input in inputs {
        add_path_to_zip(
            &mut zw,
            input,
            input.file_name().map(Path::new),
            &opts,
            sink,
        )?;
    }
    zw.finish()?;
    Ok(())
}

fn add_path_to_zip<W: Write + std::io::Seek>(
    zw: &mut zip::ZipWriter<W>,
    src: &Path,
    arc_root: Option<&Path>,
    opts: &zip::write::SimpleFileOptions,
    sink: &ProgressSink,
) -> Result<()> {
    let meta = std::fs::metadata(src)?;
    let arc_path: PathBuf = arc_root
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from(src.file_name().unwrap_or_default()));

    if meta.is_dir() {
        // Directory entry (only if non-empty path).
        let dir_name = format!("{}/", arc_path.to_string_lossy());
        if !dir_name.is_empty() && dir_name != "/" {
            zw.add_directory(dir_name, *opts)?;
        }
        for ent in std::fs::read_dir(src)? {
            let ent = ent?;
            let child_arc = arc_path.join(ent.file_name());
            add_path_to_zip(zw, &ent.path(), Some(&child_arc), opts, sink)?;
        }
    } else {
        let name = arc_path.to_string_lossy().into_owned();
        sink.send(ProgressEvent::FileStarted {
            path: name.clone(),
            size: meta.len(),
        });
        zw.start_file(&name, *opts)?;
        let mut f = BufReader::new(File::open(src)?);
        std::io::copy(&mut f, zw)?;
        sink.send(ProgressEvent::FileFinished { path: name });
    }
    Ok(())
}

fn pack_tar(
    archive: &Path,
    inputs: &[PathBuf],
    inner_algo: Option<crate::algorithm::Algorithm>,
    level: Level,
    registry: &BackendRegistry,
    sink: &ProgressSink,
) -> Result<()> {
    let file: Box<dyn Write + Send> = Box::new(BufWriter::new(File::create(archive)?));
    let writer: Box<dyn Write + Send> = match inner_algo {
        None => file,
        Some(algo) => {
            let backend = registry.pick_compressor(algo)?;
            let comp = backend.compressor(algo, level)?;
            comp.wrap_writer(file)
        }
    };

    let mut tb = tar::Builder::new(writer);
    tb.follow_symlinks(false);
    for input in inputs {
        let arc_name = input.file_name().ok_or_else(|| {
            Error::InvalidArchive(format!("input has no file name: {}", input.display()))
        })?;
        let meta = std::fs::metadata(input)?;
        if meta.is_dir() {
            tb.append_dir_all(arc_name, input)?;
        } else {
            sink.send(ProgressEvent::FileStarted {
                path: arc_name.to_string_lossy().into_owned(),
                size: meta.len(),
            });
            let mut f = File::open(input)?;
            tb.append_file(arc_name, &mut f)?;
            sink.send(ProgressEvent::FileFinished {
                path: arc_name.to_string_lossy().into_owned(),
            });
        }
    }
    let mut writer = tb.into_inner()?;
    writer.flush()?;
    // Drop writer here to flush the codec's trailer (gz footer / zstd frame end).
    drop(writer);
    let _ = inner_algo; // suppress unused warning when no codec
    Ok(())
}

// Silence unused import warning when only zip path is taken.
#[allow(dead_code)]
fn _force_read_used(_: &mut dyn Read) {}
