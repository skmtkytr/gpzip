//! indicatif consumer for `ProgressEvent`. Runs the bar/spinner on its own
//! thread so the codec pipeline never blocks on terminal I/O.

use std::thread::{self, JoinHandle};
use std::time::Duration;

use crossbeam_channel::{unbounded, Receiver};
use gpzip_core::{ProgressEvent, ProgressSink};
use indicatif::{ProgressBar, ProgressStyle};

/// One sink-and-display pair. Hand the sink to the pack/unpack/list call,
/// then call `finish()` once the call returns to drain remaining events and
/// stop the bar.
pub struct Progress {
    sink: ProgressSink,
    handle: Option<JoinHandle<()>>,
}

impl Progress {
    pub fn new(label: &str) -> Self {
        let (tx, rx) = unbounded::<ProgressEvent>();
        let label = label.to_string();
        let handle = thread::spawn(move || run_consumer(rx, label));
        Self {
            sink: ProgressSink::new(tx),
            handle: Some(handle),
        }
    }

    pub fn sink(&self) -> ProgressSink {
        self.sink.clone()
    }

    /// Drop the sink so the consumer's iteration ends, then join the thread.
    pub fn finish(mut self) {
        // Drop the local sink first so the receiver hangs up after draining.
        // Move the field out instead of replacing with default to be explicit.
        let _ = std::mem::replace(&mut self.sink, ProgressSink::noop());
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for Progress {
    fn drop(&mut self) {
        if let Some(h) = self.handle.take() {
            // Ensure sender is gone so the consumer can exit.
            let _ = std::mem::replace(&mut self.sink, ProgressSink::noop());
            let _ = h.join();
        }
    }
}

fn run_consumer(rx: Receiver<ProgressEvent>, label: String) {
    // Spinner with current-file message. Byte-level totals are coarse for
    // archive work (pack walks directories, unpack streams unknown sizes), so
    // a spinner with file count + current path is more honest than a fake
    // percentage.
    let bar = ProgressBar::new_spinner();
    bar.set_style(
        ProgressStyle::with_template("{spinner:.green} {prefix:.bold} {wide_msg} {pos} files")
            .unwrap()
            .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ "),
    );
    bar.set_prefix(label);
    bar.enable_steady_tick(Duration::from_millis(80));

    for event in rx.iter() {
        match event {
            ProgressEvent::Started { .. } => {}
            ProgressEvent::FileStarted { path, .. } => {
                bar.set_message(path);
                bar.inc(1);
            }
            ProgressEvent::FileFinished { .. } => {}
            ProgressEvent::Bytes { .. } => {}
            ProgressEvent::Finished => {
                bar.finish_with_message("done");
            }
            ProgressEvent::Error(e) => {
                bar.abandon_with_message(format!("error: {e}"));
            }
        }
    }
    // If the channel closed without a Finished/Error event, leave the bar
    // standing so the caller's own error path can be the visible signal.
    if !bar.is_finished() {
        bar.finish_and_clear();
    }
}
