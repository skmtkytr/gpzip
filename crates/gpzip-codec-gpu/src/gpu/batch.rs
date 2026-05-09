//! Two-stage GPU pipeline: a submitter thread + a completer thread.
//!
//! Each `submit` call hands a chunk to the channel and blocks until the
//! result comes back. The submitter drains incoming jobs greedily (one
//! blocking recv + a try_recv loop), assembles up to `MAX_BATCH` jobs
//! into one GPU submission, and immediately hands the in-flight handle
//! to the completer for read-back. Submission of batch N+1 overlaps with
//! the completer's wait/read of batch N — `Maintain::WaitForSubmissionIndex`
//! is per-submission, so the GPU queue can deepen to `PIPELINE_DEPTH`
//! batches without one batch's wait blocking the next.
//!
//! The earlier multi-worker attempt (each worker running its own
//! submit→`poll(Wait)`→read cycle) lost in benchmarks because `Wait` is a
//! per-device wait — every worker stalled on every other worker's
//! submissions. The submitter/completer split avoids that by using
//! `WaitForSubmissionIndex` and serialising both halves single-threaded.
//!
//! No batch timeout: when the queue has only one chunk, the submitter
//! processes it alone rather than waiting. That's the right call for
//! low-throughput periods (don't add latency for nothing) and it's never
//! *worse* than the non-pipelined path.

use std::sync::Arc;
use std::thread::{self, JoinHandle};

use crossbeam_channel::{bounded, unbounded, Receiver, Sender};

use super::lz77::Token;
use super::lz77_hash::{AsyncBatch, Lz77HashPipeline};

// 16 was tried (was 8) — neutral on wall and aggregate throughput on a
// 272 MB binmix profile (submit 1.30 → 1.28 ms/chunk, well within noise).
// The GPU pipeline tops out around 8000 chunks/sec on the test box at any
// MAX_BATCH ≥ 8 because batches don't actually fill to the cap when chunks
// arrive at the rate they do — leaving headroom is harmless.
const MAX_BATCH: usize = 16;
/// In-flight batches between submitter and completer. Bounds GPU queue
/// depth and host memory (each in-flight batch holds ~16 MiB of GPU
/// buffers in the BufferSet pool). 4 is enough to keep the GPU busy
/// while the completer drains; deeper just means more buffers parked.
const PIPELINE_DEPTH: usize = 4;

struct Job {
    input: Vec<u8>,
    response: Sender<Vec<Token>>,
}

pub struct BatchedLz77 {
    job_tx: Option<Sender<Job>>,
    submitter: Option<JoinHandle<()>>,
    completer: Option<JoinHandle<()>>,
}

impl BatchedLz77 {
    pub fn new(pipeline: Arc<Lz77HashPipeline>, window: u32) -> Self {
        let (job_tx, job_rx) = unbounded::<Job>();
        // Bounded so the submitter naturally backpressures when the
        // completer can't keep up — keeps GPU queue depth bounded.
        let (pending_tx, pending_rx) = bounded::<(AsyncBatch, Vec<Job>)>(PIPELINE_DEPTH);

        let pipe_a = Arc::clone(&pipeline);
        let submitter = thread::spawn(move || submitter_loop(job_rx, pipe_a, window, pending_tx));

        let pipe_b = Arc::clone(&pipeline);
        let completer = thread::spawn(move || completer_loop(pending_rx, pipe_b));

        Self {
            job_tx: Some(job_tx),
            submitter: Some(submitter),
            completer: Some(completer),
        }
    }

    pub fn submit(&self, input: Vec<u8>) -> Vec<Token> {
        let (tx, rx) = bounded::<Vec<Token>>(1);
        self.job_tx
            .as_ref()
            .expect("executor dropped")
            .send(Job {
                input,
                response: tx,
            })
            .expect("submitter thread gone");
        rx.recv().expect("worker dropped response")
    }
}

impl Drop for BatchedLz77 {
    fn drop(&mut self) {
        // Drop job_tx → submitter exits → drops pending_tx → completer exits.
        drop(self.job_tx.take());
        if let Some(h) = self.submitter.take() {
            let _ = h.join();
        }
        if let Some(h) = self.completer.take() {
            let _ = h.join();
        }
    }
}

fn submitter_loop(
    rx: Receiver<Job>,
    pipeline: Arc<Lz77HashPipeline>,
    window: u32,
    pending_tx: Sender<(AsyncBatch, Vec<Job>)>,
) {
    loop {
        // Block for the first job; drain whatever else is queued.
        let first = match rx.recv() {
            Ok(j) => j,
            Err(_) => return,
        };
        let mut batch = vec![first];
        while batch.len() < MAX_BATCH {
            match rx.try_recv() {
                Ok(j) => batch.push(j),
                Err(_) => break,
            }
        }

        // Build inputs view and submit to the GPU. Returns immediately
        // with an in-flight handle; doesn't wait.
        let inputs: Vec<&[u8]> = batch.iter().map(|j| j.input.as_slice()).collect();
        let async_batch = pipeline.submit_batch_async(&inputs, window);

        // Hand off to the completer. The bounded send blocks if the
        // completer is behind, which is the GPU-queue-depth bound.
        if pending_tx.send((async_batch, batch)).is_err() {
            return;
        }
    }
}

fn completer_loop(pending_rx: Receiver<(AsyncBatch, Vec<Job>)>, pipeline: Arc<Lz77HashPipeline>) {
    while let Ok((async_batch, jobs)) = pending_rx.recv() {
        let results = pipeline.collect_async(async_batch);
        for (job, tokens) in jobs.into_iter().zip(results) {
            let _ = job.response.send(tokens);
        }
    }
}
