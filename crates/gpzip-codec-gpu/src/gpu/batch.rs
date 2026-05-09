//! Background worker that batches LZ77 chunks into single GPU submissions.
//!
//! Each `submit` call hands a chunk to the worker thread and blocks until
//! the result comes back. The worker drains its incoming queue greedily
//! (one blocking recv + a try_recv loop) so any chunks queued while it
//! works on the previous batch get processed together. Cuts per-chunk
//! submit + poll overhead down to per-batch.
//!
//! ## Why one worker (not multiple)
//!
//! Profiling pointed at `device.poll(Maintain::Wait)` as ~18% of wall time
//! and the obvious fix was N parallel workers each running their own
//! submit→poll cycle. Measured at WORKERS=2 over 5 trials each, this
//! actually *lost*: rand 391→428 ms (-9%), bin 411→481 ms (-17%), rep
//! +4%. wgpu's `poll(Wait)` is a per-device wait, not per-submission, so
//! two workers serialise inside wgpu's queue with extra Mutex contention
//! on the BufferSet pool to boot. After the segmented hash + packed-token
//! shrinks (#2 and #5), GPU compute per chunk is small enough that the
//! true bottleneck moved to host-side encoding — multi-worker can't help.
//! Kept the single-worker design; the negative result is documented here
//! so a future contributor doesn't re-litigate it without new data.
//!
//! No batch timeout: when the queue has only one chunk, the worker
//! processes it alone rather than waiting. That's the right call for
//! low-throughput periods (don't add latency for nothing) and it's never
//! *worse* than the non-batched path.

use std::sync::Arc;
use std::thread::{self, JoinHandle};

use crossbeam_channel::{bounded, unbounded, Receiver, Sender};

use super::lz77::Token;
use super::lz77_hash::Lz77HashPipeline;

const MAX_BATCH: usize = 8;

struct Job {
    input: Vec<u8>,
    response: Sender<Vec<Token>>,
}

pub struct BatchedLz77 {
    sender: Option<Sender<Job>>,
    handle: Option<JoinHandle<()>>,
}

impl BatchedLz77 {
    pub fn new(pipeline: Arc<Lz77HashPipeline>, window: u32) -> Self {
        let (tx, rx): (Sender<Job>, Receiver<Job>) = unbounded();
        let handle = thread::spawn(move || worker_loop(rx, pipeline, window));
        Self {
            sender: Some(tx),
            handle: Some(handle),
        }
    }

    pub fn submit(&self, input: Vec<u8>) -> Vec<Token> {
        let (tx, rx) = bounded::<Vec<Token>>(1);
        self.sender
            .as_ref()
            .expect("executor dropped")
            .send(Job {
                input,
                response: tx,
            })
            .expect("worker thread gone");
        rx.recv().expect("worker dropped response")
    }
}

impl Drop for BatchedLz77 {
    fn drop(&mut self) {
        // Drop sender so worker exits its recv loop.
        drop(self.sender.take());
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn worker_loop(rx: Receiver<Job>, pipeline: Arc<Lz77HashPipeline>, window: u32) {
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

        // Run the batched GPU work.
        let inputs: Vec<&[u8]> = batch.iter().map(|j| j.input.as_slice()).collect();
        let results = pipeline.match_find_batch(&inputs, window);

        // Hand results back. Send failures (closed receivers) are harmless;
        // the caller may have given up.
        for (job, tokens) in batch.into_iter().zip(results) {
            let _ = job.response.send(tokens);
        }
    }
}
