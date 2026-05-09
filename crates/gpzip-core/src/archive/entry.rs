use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Entry {
    pub path: PathBuf,
    pub size: u64,
    pub is_dir: bool,
}
