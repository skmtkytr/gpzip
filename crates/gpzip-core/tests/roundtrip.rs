//! Round-trip integration tests: pack a directory, unpack, compare contents.
//!
//! Uses the CPU backend (gpzip-codec-cpu). GPU backend is exercised by its
//! own tests when available.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use gpzip_codec_cpu::CpuBackend;
use gpzip_core::archive::{list_archive, pack, unpack};
use gpzip_core::{BackendRegistry, CodecBackend, Level, ProgressSink};

fn cpu_registry() -> BackendRegistry {
    let mut r = BackendRegistry::new();
    r.push(Arc::new(CpuBackend::new()) as Arc<dyn CodecBackend>);
    r
}

/// Create a small fixture tree under `root`. Returns the file count.
fn make_fixture(root: &Path) -> u32 {
    fs::create_dir_all(root.join("nested/deep")).unwrap();
    fs::write(root.join("a.txt"), b"hello world\n").unwrap();
    fs::write(root.join("b.bin"), vec![0xAAu8; 16 * 1024]).unwrap();
    fs::write(root.join("nested/c.txt"), b"second file\n").unwrap();
    fs::write(root.join("nested/deep/d.log"), b"deeply nested\n").unwrap();
    4
}

fn assert_tree_equal(a: &Path, b: &Path) {
    let mut listing_a = walk(a);
    let mut listing_b = walk(b);
    listing_a.sort();
    listing_b.sort();
    assert_eq!(listing_a, listing_b, "tree shape mismatch");

    for rel in &listing_a {
        let pa = a.join(rel);
        let pb = b.join(rel);
        if pa.is_file() {
            assert_eq!(
                fs::read(&pa).unwrap(),
                fs::read(&pb).unwrap(),
                "content mismatch: {}",
                rel.display()
            );
        }
    }
}

fn walk(root: &Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    if !root.exists() {
        return out;
    }
    for ent in walkdir(root) {
        out.push(ent.strip_prefix(root).unwrap().to_path_buf());
    }
    out
}

fn walkdir(root: &Path) -> Vec<std::path::PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    let mut out = Vec::new();
    while let Some(p) = stack.pop() {
        if p != root {
            out.push(p.clone());
        }
        if p.is_dir() {
            for ent in fs::read_dir(&p).unwrap() {
                stack.push(ent.unwrap().path());
            }
        }
    }
    out
}

fn run_roundtrip(ext: &str) {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    let _files = make_fixture(&src);

    let archive = tmp.path().join(format!("out.{ext}"));
    let extracted = tmp.path().join("extracted");
    fs::create_dir_all(&extracted).unwrap();

    let registry = cpu_registry();
    pack(
        &archive,
        std::slice::from_ref(&src),
        Level(5),
        &registry,
        ProgressSink::noop(),
    )
    .unwrap_or_else(|e| panic!("pack {ext} failed: {e}"));

    assert!(archive.exists(), "archive {ext} not created");
    assert!(
        archive.metadata().unwrap().len() > 0,
        "archive {ext} is empty"
    );

    let entries =
        list_archive(&archive, &registry).unwrap_or_else(|e| panic!("list {ext} failed: {e}"));
    assert!(!entries.is_empty(), "{ext} listing is empty");

    unpack(&archive, &extracted, &registry, ProgressSink::noop())
        .unwrap_or_else(|e| panic!("unpack {ext} failed: {e}"));

    // The extractor restores under `extracted/src/...` because we packed `src`
    // as a top-level entry. Compare the inner subtree.
    let restored_root = extracted.join("src");
    assert!(
        restored_root.exists(),
        "expected extracted/src to exist for {ext}"
    );
    assert_tree_equal(&src, &restored_root);
}

#[test]
fn roundtrip_zip() {
    run_roundtrip("zip");
}

#[test]
fn roundtrip_tar_gz() {
    run_roundtrip("tar.gz");
}

#[test]
fn roundtrip_tar_zst() {
    run_roundtrip("tar.zst");
}

#[test]
fn roundtrip_tar_plain() {
    run_roundtrip("tar");
}

#[test]
fn unpacking_unknown_format_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let bogus = tmp.path().join("file.unknownext");
    fs::write(&bogus, b"not an archive").unwrap();
    let registry = cpu_registry();
    let r = unpack(&bogus, tmp.path(), &registry, ProgressSink::noop());
    assert!(r.is_err());
}
