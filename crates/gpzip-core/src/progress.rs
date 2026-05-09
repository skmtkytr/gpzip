use crossbeam_channel::Sender;

#[derive(Debug, Clone)]
pub enum ProgressEvent {
    Started { total_bytes: u64 },
    FileStarted { path: String, size: u64 },
    Bytes { delta: u64 },
    FileFinished { path: String },
    Finished,
    Error(String),
}

/// Sink that pipelines push progress to. CLI uses indicatif consumer; GUI
/// uses a UI-thread consumer. Sending failures (closed channel) are silently
/// dropped so the pipeline never aborts because of a missing UI listener.
#[derive(Clone, Default)]
pub struct ProgressSink {
    tx: Option<Sender<ProgressEvent>>,
}

impl ProgressSink {
    pub fn new(tx: Sender<ProgressEvent>) -> Self {
        Self { tx: Some(tx) }
    }
    pub fn noop() -> Self {
        Self { tx: None }
    }
    pub fn send(&self, ev: ProgressEvent) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(ev);
        }
    }
}
