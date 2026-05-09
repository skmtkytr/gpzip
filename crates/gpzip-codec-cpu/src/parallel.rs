//! Chunk-parallel streaming compressor.
//!
//! Splits the input into independent fixed-size chunks, dispatches each
//! chunk to a rayon worker, and writes the compressed members back to the
//! output writer in input-index order. Bounded in-flight queue keeps memory
//! use predictable.
//!
//! The chunk function must produce an *independently decodable* member
//! (gzip member, zstd frame). Concatenated members form a valid stream for
//! standard tools — gzip RFC 1952 §2.2 and zstd's multi-frame format both
//! permit this. That's how we get parallelism without inventing a new
//! container format.

use std::collections::BTreeMap;
use std::io::{self, Write};
use std::sync::Arc;

use crossbeam_channel::{bounded, Receiver, Sender};

/// Function compressing one chunk into one self-contained member.
pub type ChunkFn = Arc<dyn Fn(&[u8]) -> io::Result<Vec<u8>> + Send + Sync>;

/// Streaming Write adapter that compresses chunks in parallel.
///
/// Bytes written to it are appended to an in-progress chunk; once the chunk
/// is full it's dispatched to rayon and a fresh chunk starts. On drop, all
/// pending work is finalized and the inner writer is flushed.
pub struct ParallelChunkedWriter {
    current: Vec<u8>,
    chunk_size: usize,
    next_index: u32,
    write_index: u32,
    pending: BTreeMap<u32, Vec<u8>>,
    in_flight: usize,
    max_in_flight: usize,
    chunk_fn: ChunkFn,
    rx: Receiver<(u32, io::Result<Vec<u8>>)>,
    tx: Sender<(u32, io::Result<Vec<u8>>)>,
    output: Box<dyn Write + Send>,
    finalized: bool,
    /// First error encountered; surfaced from finalize/drop.
    sticky_err: Option<io::Error>,
}

impl ParallelChunkedWriter {
    pub fn new(
        output: Box<dyn Write + Send>,
        chunk_size: usize,
        max_in_flight: usize,
        chunk_fn: ChunkFn,
    ) -> Self {
        assert!(chunk_size > 0, "chunk_size must be > 0");
        assert!(max_in_flight > 0, "max_in_flight must be > 0");
        let (tx, rx) = bounded(max_in_flight);
        Self {
            current: Vec::with_capacity(chunk_size),
            chunk_size,
            next_index: 0,
            write_index: 0,
            pending: BTreeMap::new(),
            in_flight: 0,
            max_in_flight,
            chunk_fn,
            rx,
            tx,
            output,
            finalized: false,
            sticky_err: None,
        }
    }

    /// Stash an error so subsequent ops also fail. Avoids the race where a
    /// chunk failure is consumed by `write` but `finalize` then sees clean
    /// state and reports success.
    fn record(&mut self, e: io::Error) -> io::Error {
        let mirror = io::Error::new(e.kind(), e.to_string());
        if self.sticky_err.is_none() {
            self.sticky_err = Some(mirror);
        }
        e
    }

    fn check_sticky(&self) -> io::Result<()> {
        if let Some(e) = &self.sticky_err {
            Err(io::Error::new(e.kind(), e.to_string()))
        } else {
            Ok(())
        }
    }

    fn dispatch_current(&mut self) -> io::Result<()> {
        self.check_sticky()?;
        if self.current.is_empty() {
            return Ok(());
        }
        let idx = self.next_index;
        self.next_index += 1;
        let bytes = std::mem::replace(&mut self.current, Vec::with_capacity(self.chunk_size));
        let chunk_fn = Arc::clone(&self.chunk_fn);
        let tx = self.tx.clone();
        rayon::spawn(move || {
            let result = chunk_fn(&bytes);
            let _ = tx.send((idx, result));
        });
        self.in_flight += 1;
        if let Err(e) = self.drain_ready() {
            return Err(self.record(e));
        }
        while self.in_flight >= self.max_in_flight {
            let (i, res) = match self.rx.recv() {
                Ok(v) => v,
                Err(e) => {
                    let err = io::Error::other(format!("worker channel closed: {e}"));
                    return Err(self.record(err));
                }
            };
            self.in_flight -= 1;
            match res {
                Ok(bytes) => {
                    self.pending.insert(i, bytes);
                }
                Err(e) => return Err(self.record(e)),
            }
            if let Err(e) = self.write_pending() {
                return Err(self.record(e));
            }
        }
        Ok(())
    }

    fn drain_ready(&mut self) -> io::Result<()> {
        while let Ok((i, res)) = self.rx.try_recv() {
            self.in_flight -= 1;
            match res {
                Ok(bytes) => {
                    self.pending.insert(i, bytes);
                }
                Err(e) => return Err(e),
            }
        }
        self.write_pending()
    }

    fn write_pending(&mut self) -> io::Result<()> {
        while let Some(bytes) = self.pending.remove(&self.write_index) {
            self.output.write_all(&bytes)?;
            self.write_index += 1;
        }
        Ok(())
    }

    /// Push current buffer + drain all in-flight workers, write everything in
    /// order, flush the inner writer. Idempotent. Returns Err if any chunk
    /// has ever failed (sticky).
    pub fn finalize(&mut self) -> io::Result<()> {
        if self.finalized {
            return self.check_sticky();
        }
        self.finalized = true;
        // Try to dispatch the trailing partial chunk and drain workers, but
        // even if an early error stashed itself we must still drain the
        // outstanding workers (otherwise their tx clones keep the channel
        // alive and we'd leak threads). So we collect errors instead of early
        // returning.
        let _ = self.dispatch_current();
        while self.in_flight > 0 {
            match self.rx.recv() {
                Ok((i, res)) => {
                    self.in_flight -= 1;
                    match res {
                        Ok(bytes) => {
                            self.pending.insert(i, bytes);
                            let _ = self.write_pending().map_err(|e| self.record(e));
                        }
                        Err(e) => {
                            self.record(e);
                        }
                    }
                }
                Err(e) => {
                    let err = io::Error::other(format!("worker channel closed: {e}"));
                    self.record(err);
                    break;
                }
            }
        }
        if let Err(e) = self.output.flush() {
            self.record(e);
        }
        self.check_sticky()
    }
}

impl Write for ParallelChunkedWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if let Some(e) = self.sticky_err.take() {
            return Err(e);
        }
        let mut written = 0;
        while written < buf.len() {
            let space = self.chunk_size - self.current.len();
            let take = (buf.len() - written).min(space);
            self.current
                .extend_from_slice(&buf[written..written + take]);
            written += take;
            if self.current.len() >= self.chunk_size {
                self.dispatch_current()?;
            }
        }
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        // Intentionally NOT dispatching the current partial chunk here. tar
        // and other writers call flush() between entries; flushing partial
        // chunks would fragment compression and tank the ratio. Final flush
        // happens in `finalize()` (called from Drop).
        Ok(())
    }
}

impl Drop for ParallelChunkedWriter {
    fn drop(&mut self) {
        if let Err(e) = self.finalize() {
            // Drop can't return errors; stash for next write or just log.
            tracing::error!(target: "gpzip-codec-cpu::parallel", error = %e, "finalize on drop failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    /// Identity "compressor" — bytes are framed as `[len: u32 LE][bytes]`.
    /// Easy to verify by parsing the frames back out.
    fn identity_framer() -> ChunkFn {
        Arc::new(|bytes: &[u8]| {
            let mut out = Vec::with_capacity(4 + bytes.len());
            out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(bytes);
            Ok(out)
        })
    }

    fn parse_frames(mut data: &[u8]) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        while data.len() >= 4 {
            let len = u32::from_le_bytes(data[..4].try_into().unwrap()) as usize;
            let payload = data[4..4 + len].to_vec();
            out.push(payload);
            data = &data[4 + len..];
        }
        out
    }

    #[test]
    fn empty_input_writes_nothing() {
        let buf: Vec<u8> = Vec::new();
        {
            let _w = ParallelChunkedWriter::new(Box::new(buf.clone()), 16, 4, identity_framer());
        }
        // Output buffer was moved into the writer and dropped; can't inspect.
        // Use Vec<u8> wrapped in Arc<Mutex<>> for inspection — see below.
    }

    fn capture_output<F: FnOnce(&mut ParallelChunkedWriter)>(
        chunk_size: usize,
        max_in_flight: usize,
        chunk_fn: ChunkFn,
        body: F,
    ) -> Vec<u8> {
        use std::sync::Mutex;
        let sink: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
        struct SharedSink(Arc<Mutex<Vec<u8>>>);
        impl Write for SharedSink {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        {
            let mut w = ParallelChunkedWriter::new(
                Box::new(SharedSink(Arc::clone(&sink))),
                chunk_size,
                max_in_flight,
                chunk_fn,
            );
            body(&mut w);
            w.finalize().unwrap();
        }
        Arc::try_unwrap(sink).unwrap().into_inner().unwrap()
    }

    #[test]
    fn empty_finalize_emits_nothing() {
        let out = capture_output(16, 4, identity_framer(), |_| {});
        assert!(out.is_empty());
    }

    #[test]
    fn single_partial_chunk_emitted_on_finalize() {
        let out = capture_output(16, 4, identity_framer(), |w| {
            w.write_all(b"hello").unwrap();
        });
        let frames = parse_frames(&out);
        assert_eq!(frames, vec![b"hello".to_vec()]);
    }

    #[test]
    fn many_chunks_are_in_input_order() {
        let chunk_size = 8;
        let total = 8 * 64; // 64 chunks
        let input: Vec<u8> = (0..total as u32).map(|i| (i % 251) as u8).collect();
        let out = capture_output(chunk_size, 4, identity_framer(), |w| {
            // Write in odd small bursts to exercise buffering.
            for slice in input.chunks(7) {
                w.write_all(slice).unwrap();
            }
        });
        let frames = parse_frames(&out);
        let rebuilt: Vec<u8> = frames.into_iter().flatten().collect();
        assert_eq!(rebuilt, input, "concatenated frames must match input");
    }

    #[test]
    fn backpressure_does_not_deadlock() {
        // Tiny channel + many chunks; if backpressure logic is broken this
        // either deadlocks or drops chunks.
        let chunk_size = 4;
        let total = 4 * 200;
        let input: Vec<u8> = (0..total as u32).map(|i| i as u8).collect();
        let out = capture_output(chunk_size, 2, identity_framer(), |w| {
            w.write_all(&input).unwrap();
        });
        let frames = parse_frames(&out);
        let rebuilt: Vec<u8> = frames.into_iter().flatten().collect();
        assert_eq!(rebuilt, input);
    }

    #[test]
    fn chunk_fn_error_surfaces_through_finalize() {
        use std::sync::Mutex;
        let sink: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
        struct SharedSink(Arc<Mutex<Vec<u8>>>);
        impl Write for SharedSink {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        let failing: ChunkFn = Arc::new(|_| Err(io::Error::other("boom")));
        let mut w =
            ParallelChunkedWriter::new(Box::new(SharedSink(Arc::clone(&sink))), 4, 2, failing);
        // Push enough to trigger at least one chunk dispatch + finalize.
        let _ = w.write_all(&[0u8; 32]);
        let err = w
            .finalize()
            .expect_err("finalize should surface chunk error");
        assert!(err.to_string().contains("boom"), "got: {err}");
    }

    #[test]
    fn gzip_chunked_output_is_valid_gzip() {
        use flate2::read::MultiGzDecoder;
        use flate2::write::GzEncoder;
        use flate2::Compression;

        let level = 5u32;
        let gz: ChunkFn = Arc::new(move |bytes: &[u8]| {
            let mut e = GzEncoder::new(Vec::new(), Compression::new(level));
            e.write_all(bytes)?;
            e.finish()
        });

        let chunk_size = 1024;
        let total = chunk_size * 7 + 100; // 7 full + 1 partial chunks
        let input: Vec<u8> = (0..total).map(|i| (i % 251) as u8).collect();

        let out = capture_output(chunk_size, 4, gz, |w| {
            w.write_all(&input).unwrap();
        });

        let mut decoded = Vec::new();
        MultiGzDecoder::new(&out[..])
            .read_to_end(&mut decoded)
            .expect("MultiGzDecoder must read concatenated members");
        assert_eq!(decoded, input);
    }

    #[test]
    fn zstd_chunked_output_is_valid_zstd() {
        let level = 3;
        let zs: ChunkFn = Arc::new(move |bytes: &[u8]| {
            let mut e = zstd::stream::write::Encoder::new(Vec::new(), level)?;
            e.write_all(bytes)?;
            e.finish()
        });

        let chunk_size = 2048;
        let total = chunk_size * 5 + 7;
        let input: Vec<u8> = (0..total).map(|i| (i % 211) as u8).collect();

        let out = capture_output(chunk_size, 3, zs, |w| {
            w.write_all(&input).unwrap();
        });

        // zstd's read::Decoder handles multi-frame streams natively.
        let mut decoded = Vec::new();
        zstd::stream::read::Decoder::new(&out[..])
            .unwrap()
            .read_to_end(&mut decoded)
            .expect("zstd Decoder must read concatenated frames");
        assert_eq!(decoded, input);
    }
}
