//! C-1: GPU-side strict-greedy LZ77 walk.
//!
//! Takes per-position tokens (the output of `Lz77HashPipeline::match_find`)
//! and produces the walked, non-overlapping token stream entirely on
//! the GPU. The PoC win isn't speed — single-thread serial walk on the
//! GPU is comparable to or slightly slower than the host walk — it's
//! that walked tokens never have to round-trip through the host before
//! the encoder consumes them. Once both walk and encode are GPU-side,
//! `match_find → walk → encode` can run as one batched dispatch (D-7
//! / sub-agent's §4-A).
//!
//! Strict greedy gives up the host walk's lazy peek (drop current
//! match if the next position has a strictly longer one), which costs
//! ~1-3% compression ratio on text-heavy input. Acceptable for a PoC;
//! a parallel lazy walk is research-track.
//!
//! Status: standalone PoC. Not yet wired into the chunk_fn — the
//! integration with `HuffmanEmitV2Pipeline` (sharing the walked
//! buffer) is the next piece.

#![allow(dead_code)]

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::context::GpuContext;
use super::lz77::Token;

const SHADER: &str = include_str!("walk.wgsl");
const BLOCK_SHADER: &str = include_str!("walk_block.wgsl");

/// Block size W for the block-parallel walk. Must match the
/// workgroup_size declared in walk_block.wgsl. 128 keeps the per-block
/// serial work small while limiting per-block memory (exit_table is
/// W u32 entries per block).
pub const BLOCK_SIZE: u32 = 128;

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Params {
    n_positions: u32,
    _pad: [u32; 3],
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct BlockParams {
    n_positions: u32,
    n_blocks: u32,
    block_size: u32,
    _pad: u32,
}

pub struct WalkPipeline {
    ctx: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl WalkPipeline {
    pub fn new(ctx: Arc<GpuContext>) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpzip-walk-shader"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpzip-walk-bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                    storage_entry(2, false),
                    uniform_entry(3),
                ],
            });
        let pl_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpzip-walk-pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gpzip-walk-pipeline"),
                layout: Some(&pl_layout),
                module: &module,
                entry_point: "walk_serial",
            });
        Self { ctx, pipeline, bgl }
    }

    /// Run the GPU walk on a host-supplied per-position token stream.
    /// Returns the walked tokens (packed u32 form, ready to feed the
    /// emit pipeline directly). Used by tests + bench.
    pub fn walk(&self, per_position_packed: &[u32]) -> Vec<u32> {
        let n = per_position_packed.len() as u32;
        if n == 0 {
            return Vec::new();
        }

        let buf_input = self
            .ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpzip-walk-input"),
                contents: bytemuck::cast_slice(per_position_packed),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let walked_bytes = (per_position_packed.len() * 4) as u64;
        let buf_walked = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpzip-walk-walked"),
            size: walked_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_walked = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpzip-walk-staging-walked"),
            size: walked_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_count = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpzip-walk-count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_count = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpzip-walk-staging-count"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = Params {
            n_positions: n,
            _pad: [0; 3],
        };
        let buf_params = self
            .ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpzip-walk-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpzip-walk-bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf_input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf_walked.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buf_count.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buf_params.as_entire_binding(),
                    },
                ],
            });
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpzip-walk-enc"),
            });
        encoder.clear_buffer(&buf_count, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpzip-walk-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&buf_walked, 0, &staging_walked, 0, walked_bytes);
        encoder.copy_buffer_to_buffer(&buf_count, 0, &staging_count, 0, 4);
        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        // Read count first, then trim walked.
        let count_slice = staging_count.slice(..);
        let walked_slice = staging_walked.slice(..);
        let (tx_c, rx_c) = std::sync::mpsc::channel();
        let (tx_w, rx_w) = std::sync::mpsc::channel();
        count_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx_c.send(r);
        });
        walked_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx_w.send(r);
        });
        self.ctx.device.poll(wgpu::Maintain::Wait);
        rx_c.recv().unwrap().expect("count map failed");
        rx_w.recv().unwrap().expect("walked map failed");

        let count = {
            let view = count_slice.get_mapped_range();
            let n = u32::from_le_bytes(view[..4].try_into().unwrap()) as usize;
            drop(view);
            staging_count.unmap();
            n
        };
        let view = walked_slice.get_mapped_range();
        let walked: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&view)[..count].to_vec();
        drop(view);
        staging_walked.unmap();
        walked
    }
}

/// Block-parallel strict-greedy walk. 3 dispatches in one submit:
/// per-block summary (parallel), serial chain through blocks (single
/// thread), per-block emit (parallel). Walked output is bit-identical
/// to the C-1 serial shader and to host_strict_greedy; only the
/// algorithm shape differs.
pub struct BlockWalkPipeline {
    ctx: Arc<GpuContext>,
    bgl: wgpu::BindGroupLayout,
    p_summary: wgpu::ComputePipeline,
    p_chain: wgpu::ComputePipeline,
    p_emit: wgpu::ComputePipeline,
}

impl BlockWalkPipeline {
    pub fn new(ctx: Arc<GpuContext>) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpzip-walk-block-shader"),
                source: wgpu::ShaderSource::Wgsl(BLOCK_SHADER.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpzip-walk-block-bgl"),
                entries: &[
                    storage_entry(0, true),  // per_position
                    storage_entry(1, false), // exit_table
                    storage_entry(2, false), // count_table
                    storage_entry(3, false), // actual_entry
                    storage_entry(4, false), // token_offsets
                    storage_entry(5, false), // walked
                    storage_entry(6, false), // walked_count (atomic)
                    uniform_entry(7),        // params
                ],
            });
        let pl_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpzip-walk-block-pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
        let mk_pipeline = |entry: &str, label: &str| {
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pl_layout),
                    module: &module,
                    entry_point: entry,
                })
        };
        Self {
            p_summary: mk_pipeline("block_summary", "walk-block-summary"),
            p_chain: mk_pipeline("chain_blocks", "walk-block-chain"),
            p_emit: mk_pipeline("block_emit", "walk-block-emit"),
            ctx,
            bgl,
        }
    }

    pub fn walk(&self, per_position_packed: &[u32]) -> Vec<u32> {
        let n = per_position_packed.len() as u32;
        if n == 0 {
            return Vec::new();
        }
        let n_blocks = n.div_ceil(BLOCK_SIZE);

        let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE;
        let storage_dst = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let storage_src = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let storage_src_dst = storage_dst | wgpu::BufferUsages::COPY_SRC;

        let table_bytes = (n_blocks as u64) * (BLOCK_SIZE as u64) * 4;
        let buf_input = self
            .ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("walk-block-input"),
                contents: bytemuck::cast_slice(per_position_packed),
                usage: storage,
            });
        let buf_exit = mk("walk-block-exit", table_bytes, storage);
        let buf_count = mk("walk-block-count", table_bytes, storage);
        let buf_entry = mk("walk-block-entry", (n_blocks as u64) * 4, storage);
        let buf_offsets = mk("walk-block-offsets", (n_blocks as u64) * 4, storage);
        let walked_bytes = (per_position_packed.len() * 4) as u64;
        let buf_walked = mk("walk-block-walked", walked_bytes, storage_src);
        let staging_walked = mk(
            "walk-block-staging-walked",
            walked_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let buf_count_atomic = mk("walk-block-count-atomic", 4, storage_src_dst);
        let staging_count = mk(
            "walk-block-staging-count",
            4,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        let params = BlockParams {
            n_positions: n,
            n_blocks,
            block_size: BLOCK_SIZE,
            _pad: 0,
        };
        let buf_params = self
            .ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("walk-block-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("walk-block-bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf_input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf_exit.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buf_count.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buf_entry.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: buf_offsets.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: buf_walked.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: buf_count_atomic.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: buf_params.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("walk-block-enc"),
            });
        encoder.clear_buffer(&buf_count_atomic, 0, None);
        // Pass 1: block summary (workgroup_size = BLOCK_SIZE; one workgroup per block).
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("walk-block-summary"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.p_summary);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n_blocks, 1, 1);
        }
        // Pass 2: serial chain (single thread).
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("walk-block-chain"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.p_chain);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        // Pass 3: per-block emit (one workgroup per block, 1 thread each).
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("walk-block-emit"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.p_emit);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n_blocks, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&buf_walked, 0, &staging_walked, 0, walked_bytes);
        encoder.copy_buffer_to_buffer(&buf_count_atomic, 0, &staging_count, 0, 4);
        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        let count_slice = staging_count.slice(..);
        let walked_slice = staging_walked.slice(..);
        let (tx_c, rx_c) = std::sync::mpsc::channel();
        let (tx_w, rx_w) = std::sync::mpsc::channel();
        count_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx_c.send(r);
        });
        walked_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx_w.send(r);
        });
        self.ctx.device.poll(wgpu::Maintain::Wait);
        rx_c.recv().unwrap().expect("count map failed");
        rx_w.recv().unwrap().expect("walked map failed");
        let count = {
            let view = count_slice.get_mapped_range();
            let n = u32::from_le_bytes(view[..4].try_into().unwrap()) as usize;
            drop(view);
            staging_count.unmap();
            n
        };
        let view = walked_slice.get_mapped_range();
        let walked: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&view)[..count].to_vec();
        drop(view);
        staging_walked.unmap();
        walked
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Pack a host `Token` slice into the per-position u32 representation
/// the walk shader expects.
pub fn pack_per_position(tokens: &[Token]) -> Vec<u32> {
    tokens
        .iter()
        .map(|t| (t.length << 16) | (t.distance & 0xFFFF))
        .collect()
}

/// Unpack the walk shader's output back to host `Token`s.
pub fn unpack_walked(packed: &[u32]) -> Vec<Token> {
    packed
        .iter()
        .map(|&w| Token {
            length: w >> 16,
            distance: w & 0xFFFF,
        })
        .collect()
}

/// Reference host strict-greedy walk for tests. Same as the GPU shader
/// but in Rust — no lazy peek.
pub fn host_strict_greedy(per_position: &[Token]) -> Vec<Token> {
    let mut out = Vec::new();
    let mut p = 0usize;
    while p < per_position.len() {
        let t = per_position[p];
        out.push(t);
        if t.is_literal() {
            p += 1;
        } else {
            p += t.length as usize;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_pipeline() -> Option<WalkPipeline> {
        let ctx = GpuContext::try_init().ok()?;
        Some(WalkPipeline::new(Arc::new(ctx)))
    }

    fn try_block_pipeline() -> Option<BlockWalkPipeline> {
        let ctx = GpuContext::try_init().ok()?;
        Some(BlockWalkPipeline::new(Arc::new(ctx)))
    }

    fn assert_walk_matches(per_position: &[Token]) {
        let Some(p) = try_pipeline() else {
            eprintln!("no GPU — skipping walk test");
            return;
        };
        let packed = pack_per_position(per_position);
        let gpu_walked_packed = p.walk(&packed);
        let gpu_walked = unpack_walked(&gpu_walked_packed);
        let host_walked = host_strict_greedy(per_position);
        assert_eq!(
            gpu_walked,
            host_walked,
            "GPU strict-greedy walk should match host on {} positions",
            per_position.len()
        );
    }

    fn assert_block_walk_matches(per_position: &[Token]) {
        let Some(p) = try_block_pipeline() else {
            eprintln!("no GPU — skipping block walk test");
            return;
        };
        let packed = pack_per_position(per_position);
        let gpu_walked_packed = p.walk(&packed);
        let gpu_walked = unpack_walked(&gpu_walked_packed);
        let host_walked = host_strict_greedy(per_position);
        assert_eq!(
            gpu_walked,
            host_walked,
            "block-parallel GPU walk should match host strict-greedy on {} positions",
            per_position.len()
        );
    }

    #[test]
    fn block_empty() {
        assert_block_walk_matches(&[]);
    }

    #[test]
    fn block_all_literals() {
        let t: Vec<Token> = (0..32u8).map(Token::literal).collect();
        assert_block_walk_matches(&t);
    }

    #[test]
    fn block_single_back_ref() {
        let mut t: Vec<Token> = (0..8u8).map(Token::literal).collect();
        t.push(Token::back_ref(5, 3));
        for i in 9..16u8 {
            t.push(Token::literal(i));
        }
        assert_block_walk_matches(&t);
    }

    /// Many blocks (4 KiB-ish positions = 32+ blocks at W=128). Exercises
    /// the chain across blocks.
    #[test]
    fn block_many_blocks() {
        let mut t = Vec::new();
        for i in 0..4096u32 {
            if i % 13 == 0 && i > 0 {
                t.push(Token::back_ref(7, 1));
            } else {
                t.push(Token::literal((i & 0xff) as u8));
            }
        }
        assert_block_walk_matches(&t);
    }

    /// Back-ref that crosses a block boundary. The walk's exit_table
    /// for the previous block records `cur - block_start` which can be
    /// > block_size; pass 2's chain logic must skip the block whose
    /// start is < cur.
    #[test]
    fn block_back_ref_crosses_boundary() {
        let mut t = Vec::new();
        for i in 0..200u32 {
            t.push(Token::literal((i & 0xff) as u8));
        }
        // Position 200 is a back-ref that lands at 200 + 50 = 250
        // (skipping over part of block 1 starting at 128).
        t.push(Token::back_ref(50, 100));
        for i in 251..400u32 {
            t.push(Token::literal((i & 0xff) as u8));
        }
        assert_block_walk_matches(&t);
    }

    /// Stress test: 131K positions (matches the 128 KiB chunk used
    /// elsewhere in the bench). All literals so every position is
    /// emitted; exercises the full chain through ~1024 blocks.
    #[test]
    fn block_realistic_size_all_literals() {
        let t: Vec<Token> = (0..131072u32)
            .map(|i| Token::literal((i & 0xff) as u8))
            .collect();
        assert_block_walk_matches(&t);
    }

    #[test]
    fn empty() {
        assert_walk_matches(&[]);
    }

    #[test]
    fn all_literals() {
        let t: Vec<Token> = (0..32u8).map(Token::literal).collect();
        assert_walk_matches(&t);
    }

    #[test]
    fn single_back_ref() {
        let mut t: Vec<Token> = (0..8u8).map(Token::literal).collect();
        t.push(Token::back_ref(5, 3));
        // After (5, 3) at position 8, next walked position is 13. Pad
        // the per-position array to that length so the walk has somewhere
        // to land.
        for i in 9..16u8 {
            t.push(Token::literal(i));
        }
        assert_walk_matches(&t);
    }

    #[test]
    fn long_match_skips_positions() {
        // Position 0 is a back-ref of length 100; the walk should land
        // at position 100, skipping 1..99 entirely.
        let mut t = vec![Token::back_ref(100, 50)];
        for i in 1..200u32 {
            t.push(Token::literal((i & 0xff) as u8));
        }
        assert_walk_matches(&t);
    }

    #[test]
    fn realistic_mix() {
        // Mix of literals and varying-length matches.
        let mut t = Vec::new();
        for i in 0..256u32 {
            if i % 7 == 0 && i + 10 < 256 {
                t.push(Token::back_ref(5, 1));
            } else {
                t.push(Token::literal((i & 0xff) as u8));
            }
        }
        assert_walk_matches(&t);
    }

    /// Wall-time A/B for walk alone: GPU single-thread serial vs host
    /// strict-greedy. The GPU number includes the full per-call
    /// dispatch overhead (buffer create + submit + poll + readback) —
    /// the architectural win from C only materialises if walk is
    /// chained with match_find + encode in one batched submission, so
    /// this single-call bench is the *worst* case for GPU walk. Useful
    /// as a sanity floor.
    #[test]
    #[ignore]
    fn bench_walk_vs_host() {
        use crate::gpu::lz77_hash::{Lz77HashPipeline, DEFAULT_WINDOW};
        use std::time::Instant;

        let Some(walk_p) = try_pipeline() else {
            eprintln!("no GPU — skipping bench");
            return;
        };
        let Some(block_p) = try_block_pipeline() else {
            return;
        };
        let ctx = GpuContext::try_init().ok().unwrap();
        let lz77 = Lz77HashPipeline::new(Arc::new(ctx));

        let chunk = 128 * 1024usize;
        let workloads: Vec<(&str, Vec<u8>)> = vec![
            ("rand", {
                let mut v = Vec::with_capacity(chunk);
                let mut x = 0xdeadbeefu32;
                for _ in 0..chunk {
                    x = x.wrapping_mul(2654435761).wrapping_add(1);
                    v.push((x >> 8) as u8);
                }
                v
            }),
            ("rep", {
                let pat = b"the quick brown fox jumps over the lazy dog 12345 ";
                let mut v = Vec::with_capacity(chunk);
                while v.len() < chunk {
                    v.extend_from_slice(pat);
                }
                v.truncate(chunk);
                v
            }),
            ("bin", {
                let seed = std::fs::read("/usr/bin/bash").unwrap_or_else(|_| vec![0u8; 4096]);
                let mut v = Vec::with_capacity(chunk);
                while v.len() < chunk {
                    v.extend_from_slice(&seed);
                }
                v.truncate(chunk);
                v
            }),
        ];

        eprintln!();
        eprintln!(
            "{:<6} {:>9} {:>9} {:>9} {:>9} {:>9}  {:>9} {:>9}",
            "wkld", "n_pos", "n_walked", "host_ms", "serial_ms", "block_ms", "ser/host", "blk/host"
        );
        eprintln!("{}", "-".repeat(85));

        for (name, data) in &workloads {
            let raw = lz77.match_find(data, DEFAULT_WINDOW);
            let packed = pack_per_position(&raw);
            let n_pos = raw.len();

            // Warm up.
            for _ in 0..2 {
                let _ = host_strict_greedy(&raw);
                let _ = walk_p.walk(&packed);
                let _ = block_p.walk(&packed);
            }

            let iters = 32;

            let t = Instant::now();
            let mut host_walked = Vec::new();
            for _ in 0..iters {
                host_walked = host_strict_greedy(&raw);
            }
            let host_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

            let t = Instant::now();
            let mut serial_walked = Vec::new();
            for _ in 0..iters {
                serial_walked = walk_p.walk(&packed);
            }
            let serial_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

            let t = Instant::now();
            let mut block_walked = Vec::new();
            for _ in 0..iters {
                block_walked = block_p.walk(&packed);
            }
            let block_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

            assert_eq!(
                serial_walked.len(),
                host_walked.len(),
                "{name}: serial GPU walked count must match host"
            );
            assert_eq!(
                block_walked.len(),
                host_walked.len(),
                "{name}: block-parallel GPU walked count must match host"
            );
            let n_walked = host_walked.len();

            eprintln!(
                "{:<6} {:>9} {:>9} {:>7.3}ms {:>7.3}ms {:>7.3}ms  {:>7.2}x {:>7.2}x",
                name,
                n_pos,
                n_walked,
                host_ms,
                serial_ms,
                block_ms,
                serial_ms / host_ms,
                block_ms / host_ms
            );
        }

        eprintln!();
        eprintln!("Notes:");
        eprintln!("  host_ms   = `host_strict_greedy` (Rust loop, single thread)");
        eprintln!("  serial_ms = WalkPipeline::walk            (C-1: 1 GPU thread)");
        eprintln!("  block_ms  = BlockWalkPipeline::walk       (C-2: 3-pass block-parallel)");
        eprintln!("  ratios    = gpu / host (lower is better for GPU)");
    }
}
