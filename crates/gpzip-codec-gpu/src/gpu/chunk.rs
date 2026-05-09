//! Chunk-Member Profile (CMP) types. See cozip's
//! `docs/gpu-deflate-chunk-pipeline.md` for the rationale.
//!
//! Skeleton only — actual scheduling lands with the shader work (task #10).
#![allow(dead_code)]

#[derive(Debug, Clone)]
pub struct ChunkJob {
    pub index: u32,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkBackend {
    Cpu,
    Gpu,
}

#[derive(Debug, Clone)]
pub struct ChunkMember {
    pub index: u32,
    pub backend: ChunkBackend,
    pub raw_len: u32,
    pub payload: Vec<u8>,
    pub crc32: u32,
}

/// Default chunk size. Cozip uses 1–8 MiB; we start at 2 MiB.
pub const DEFAULT_HOST_CHUNK_SIZE: usize = 2 * 1024 * 1024;

/// Split a contiguous input into independent chunks.
pub fn plan_chunks(input: &[u8], chunk_size: usize) -> Vec<ChunkJob> {
    if input.is_empty() {
        return Vec::new();
    }
    let mut jobs = Vec::with_capacity(input.len().div_ceil(chunk_size));
    for (idx, slice) in input.chunks(chunk_size).enumerate() {
        jobs.push(ChunkJob {
            index: idx as u32,
            bytes: slice.to_vec(),
        });
    }
    jobs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_yields_no_chunks() {
        assert!(plan_chunks(&[], 1024).is_empty());
    }

    #[test]
    fn chunks_are_indexed_in_order() {
        let input = vec![0u8; 5000];
        let jobs = plan_chunks(&input, 2048);
        assert_eq!(jobs.len(), 3);
        assert_eq!(jobs[0].index, 0);
        assert_eq!(jobs[1].index, 1);
        assert_eq!(jobs[2].index, 2);
        assert_eq!(jobs[0].bytes.len(), 2048);
        assert_eq!(jobs[2].bytes.len(), 5000 - 4096);
    }
}
