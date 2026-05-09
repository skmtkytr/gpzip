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

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Params {
    n_positions: u32,
    _pad: [u32; 3],
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
            "{:<6} {:>9} {:>9} {:>9} {:>9}  {:>9}",
            "wkld", "n_pos", "n_walked", "host_ms", "gpu_ms", "ratio"
        );
        eprintln!("{}", "-".repeat(70));

        for (name, data) in &workloads {
            let raw = lz77.match_find(data, DEFAULT_WINDOW);
            let packed = pack_per_position(&raw);
            let n_pos = raw.len();

            // Warm up.
            for _ in 0..2 {
                let _ = host_strict_greedy(&raw);
                let _ = walk_p.walk(&packed);
            }

            let iters = 32;

            let t = Instant::now();
            let mut host_walked = Vec::new();
            for _ in 0..iters {
                host_walked = host_strict_greedy(&raw);
            }
            let host_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

            let t = Instant::now();
            let mut gpu_walked = Vec::new();
            for _ in 0..iters {
                gpu_walked = walk_p.walk(&packed);
            }
            let gpu_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

            assert_eq!(
                gpu_walked.len(),
                host_walked.len(),
                "{name}: GPU walked count must match host strict greedy"
            );
            let n_walked = host_walked.len();

            eprintln!(
                "{:<6} {:>9} {:>9} {:>7.3}ms {:>7.3}ms  {:>6.2}x",
                name,
                n_pos,
                n_walked,
                host_ms,
                gpu_ms,
                gpu_ms / host_ms
            );
        }

        eprintln!();
        eprintln!("Notes:");
        eprintln!("  host_ms = `host_strict_greedy` (Rust loop, single thread)");
        eprintln!("  gpu_ms  = WalkPipeline::walk (single-thread shader + per-call buffer setup)");
        eprintln!("  ratio   = gpu_ms / host_ms (lower is better for GPU)");
    }
}
