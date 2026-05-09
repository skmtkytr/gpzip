//! Parallel gzip decompression for gpzip-written archives.
//!
//! gpzip writes a `.tar.gz` as a sequence of independent gzip members,
//! each with the same fixed 10-byte header (no FNAME / FEXTRA / FCOMMENT,
//! per `gpzip-codec-gpu/src/gpu/deflate.rs::gzip_wrap` and the matching
//! flate2 path). We scan the compressed bytes for that header to recover
//! member boundaries, then decode each member in parallel via rayon.
//!
//! For non-gpzip gzip files (system `gzip`, `pigz` with default flags,
//! anything that includes FNAME) the magic scan finds at most one member
//! and we fall back to the serial `MultiGzDecoder`. So the optimisation
//! only kicks in for files we wrote ourselves — but that's exactly the
//! interesting case for the `gpzip x` command on a `gpzip a` output.
//!
//! Memory cost: the whole compressed input plus the whole decompressed
//! output sit in RAM at once. On 16 GiB+ desktops this is fine for
//! gigabyte-scale archives; for very large or memory-constrained inputs
//! the caller should stick with the serial path.

use std::io::{self, Read};

use flate2::read::{GzDecoder, MultiGzDecoder};
use rayon::prelude::*;

/// 10-byte gpzip-written gzip member header. Matches what
/// `gpzip-codec-gpu::gpu::deflate::gzip_wrap` emits and what the CPU
/// flate2 path emits at level 5 (default) — both produce headers without
/// optional fields in our pipeline.
const GPZIP_HEADER: &[u8] = &[0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff];
/// Variant emitted by flate2's `GzEncoder` (used in the CPU compress path)
/// — same magic + CM, but XFL is 0x04 (fastest? no, flate2 sets it based
/// on level) and OS is 0xff. Real-world flate2 writes 0x1f 0x8b 0x08 0x00
/// 0x00 0x00 0x00 0x00 0x00 0xff for level 5 too. Both variants are
/// covered by GPZIP_HEADER above; this comment documents the equivalence.
fn find_member_starts(data: &[u8]) -> Vec<usize> {
    let mut starts = Vec::new();
    if data.len() < GPZIP_HEADER.len() {
        return starts;
    }
    // Linear scan. memchr-style optimisations would help for huge files
    // but the typical case (≤ 1 GiB compressed) finishes in a few ms.
    let n = data.len();
    let mut i = 0;
    while i + GPZIP_HEADER.len() <= n {
        if data[i] == 0x1f
            && data[i + 1] == 0x8b
            && data[i + 2] == 0x08
            && &data[i..i + GPZIP_HEADER.len()] == GPZIP_HEADER
        {
            starts.push(i);
            // Skip past the header — we won't find another start inside
            // the header bytes, and the DEFLATE payload starts here.
            i += GPZIP_HEADER.len();
        } else {
            i += 1;
        }
    }
    starts
}

/// Decode a fully-buffered gzip stream into a Vec, using parallel
/// decoders if we can identify multiple member boundaries.
pub fn parallel_decompress(compressed: &[u8]) -> io::Result<Vec<u8>> {
    if compressed.is_empty() {
        return Ok(Vec::new());
    }
    let starts = find_member_starts(compressed);

    // Need at least two members for parallelism to win after slurp cost.
    if starts.len() < 2 {
        let mut out = Vec::new();
        MultiGzDecoder::new(compressed).read_to_end(&mut out)?;
        return Ok(out);
    }

    // Slice compressed bytes into per-member ranges, decode each with a
    // single-member GzDecoder. If any decode fails (e.g. a false-positive
    // header inside DEFLATE bytes — astronomically unlikely for the
    // 10-byte pattern but we don't *prove* it), fall back to the serial
    // decoder over the whole stream.
    let n = starts.len();
    let parts: Result<Vec<Vec<u8>>, _> = (0..n)
        .into_par_iter()
        .map(|i| {
            let start = starts[i];
            let end = if i + 1 < n {
                starts[i + 1]
            } else {
                compressed.len()
            };
            let mut out = Vec::new();
            GzDecoder::new(&compressed[start..end]).read_to_end(&mut out)?;
            Ok::<_, io::Error>(out)
        })
        .collect();

    match parts {
        Ok(parts) => {
            let total: usize = parts.iter().map(|p| p.len()).sum();
            let mut out = Vec::with_capacity(total);
            for p in parts {
                out.extend_from_slice(&p);
            }
            Ok(out)
        }
        Err(_) => {
            // Boundary detection mis-fired; fall back to serial.
            let mut out = Vec::new();
            MultiGzDecoder::new(compressed).read_to_end(&mut out)?;
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    fn gz_one(bytes: &[u8]) -> Vec<u8> {
        let mut e = GzEncoder::new(Vec::new(), Compression::new(5));
        e.write_all(bytes).unwrap();
        e.finish().unwrap()
    }

    #[test]
    fn single_member_falls_back_to_serial() {
        let input = b"hello world hello world hello world".repeat(100);
        let compressed = gz_one(&input);
        let starts = find_member_starts(&compressed);
        assert_eq!(starts.len(), 1, "single gpzip-style member");
        let decoded = parallel_decompress(&compressed).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn multiple_members_decode_in_parallel() {
        let mut compressed = Vec::new();
        let mut expected = Vec::new();
        for i in 0..16 {
            let chunk = format!("chunk {i} ").repeat(2000);
            expected.extend_from_slice(chunk.as_bytes());
            compressed.extend_from_slice(&gz_one(chunk.as_bytes()));
        }
        let starts = find_member_starts(&compressed);
        assert_eq!(starts.len(), 16, "should find 16 member starts");
        let decoded = parallel_decompress(&compressed).unwrap();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn empty_input() {
        let decoded = parallel_decompress(&[]).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn non_gpzip_header_falls_back() {
        // Construct a gzip with FNAME flag set — won't match GPZIP_HEADER.
        // flate2's GzBuilder lets us add a filename, which sets FNAME (0x08).
        use flate2::GzBuilder;
        let payload = b"some data".repeat(1000);
        let mut buf = Vec::new();
        GzBuilder::new()
            .filename("foo.txt")
            .write(&mut buf, Compression::new(5))
            .write_all(&payload)
            .unwrap();
        // The buf was written-into; GzBuilder's write returned a Write that
        // didn't auto-finish on drop above. Need explicit finish.
        // Simpler: just verify the header doesn't match ours and
        // parallel_decompress still works via fallback.
        let mut e = GzBuilder::new()
            .filename("foo.txt")
            .write(Vec::new(), Compression::new(5));
        e.write_all(&payload).unwrap();
        let compressed = e.finish().unwrap();
        let starts = find_member_starts(&compressed);
        assert_eq!(
            starts.len(),
            0,
            "FNAME-bearing header shouldn't match GPZIP_HEADER"
        );
        let decoded = parallel_decompress(&compressed).unwrap();
        assert_eq!(decoded, payload);
    }
}
