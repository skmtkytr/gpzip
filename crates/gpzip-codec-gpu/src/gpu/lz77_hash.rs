//! Hash-table LZ77 on the GPU. Two-pass design:
//!
//! 1. **Build pass** — every input position writes its 3-byte hash into a
//!    hash table slot. `atomicMin` keeps the oldest position seen for each
//!    bucket; later writers can't displace an earlier one. This concedes
//!    "best match" quality (ideal LZ77 wants the *closest* prior position)
//!    in exchange for a lock-free O(1) build.
//!
//! 2. **Lookup pass** — every position reads its slot, verifies the 3-byte
//!    hit isn't a hash collision, then extends the match forward.
//!
//! Output token format matches `lz77.rs`: one (length, distance) per input
//! byte. The host's `greedy_walk` and the rest of the gzip pipeline are
//! shared between brute-force and hash variants.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::context::GpuContext;
use super::lz77::Token;

const BUILD_SHADER: &str = include_str!("lz77_hash_build.wgsl");
const LOOKUP_SHADER: &str = include_str!("lz77_hash_lookup.wgsl");

/// 2^HASH_BITS slots. 16 → 64K slots, ~256 KiB per chunk hash table.
pub const HASH_BITS: u32 = 16;
const HASH_SIZE: usize = 1 << HASH_BITS;

pub const MIN_MATCH: u32 = 3;
pub const MAX_MATCH: u32 = 258;
pub const DEFAULT_WINDOW: u32 = 32 * 1024;

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Params {
    input_len: u32,
    hash_bits: u32,
    window: u32,
    min_match: u32,
    max_match: u32,
    _pad: [u32; 3],
}

pub struct Lz77HashPipeline {
    ctx: Arc<GpuContext>,
    build_pipeline: wgpu::ComputePipeline,
    lookup_pipeline: wgpu::ComputePipeline,
    build_layout: wgpu::BindGroupLayout,
    lookup_layout: wgpu::BindGroupLayout,
}

impl Lz77HashPipeline {
    pub fn new(ctx: Arc<GpuContext>) -> Self {
        let build_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpzip-lz77-hash-build"),
                source: wgpu::ShaderSource::Wgsl(BUILD_SHADER.into()),
            });
        let lookup_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpzip-lz77-hash-lookup"),
                source: wgpu::ShaderSource::Wgsl(LOOKUP_SHADER.into()),
            });

        // Build: input(read) | hash_table(atomic read_write) | params(uniform)
        let build_layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpzip-lz77-hash-build-bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                    uniform_entry(2),
                ],
            });
        // Lookup: input(read) | hash_table(read) | tokens(read_write) | params
        let lookup_layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpzip-lz77-hash-lookup-bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, false),
                    uniform_entry(3),
                ],
            });

        let build_pl_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpzip-lz77-hash-build-pl"),
                bind_group_layouts: &[&build_layout],
                push_constant_ranges: &[],
            });
        let lookup_pl_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpzip-lz77-hash-lookup-pl"),
                bind_group_layouts: &[&lookup_layout],
                push_constant_ranges: &[],
            });

        let build_pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gpzip-lz77-hash-build-pipeline"),
                layout: Some(&build_pl_layout),
                module: &build_module,
                entry_point: "build",
            });
        let lookup_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("gpzip-lz77-hash-lookup-pipeline"),
                    layout: Some(&lookup_pl_layout),
                    module: &lookup_module,
                    entry_point: "lookup",
                });

        Self {
            ctx,
            build_pipeline,
            lookup_pipeline,
            build_layout,
            lookup_layout,
        }
    }

    /// Per-position LZ77. Same output shape as the brute-force pipeline.
    pub fn match_find(&self, input: &[u8], window: u32) -> Vec<Token> {
        let n = input.len() as u32;
        if n == 0 {
            return Vec::new();
        }

        let padded = input.len().next_multiple_of(4);
        let mut input_padded = Vec::with_capacity(padded);
        input_padded.extend_from_slice(input);
        input_padded.resize(padded, 0);

        let input_buffer = self
            .ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpzip-lz77-hash-input"),
                contents: &input_padded,
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Hash table initialized to all 0xFF (= u32::MAX), so atomicMin
        // accepts any first writer.
        let hash_table_bytes = vec![0xffu8; HASH_SIZE * 4];
        let hash_table = self
            .ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpzip-lz77-hash-table"),
                contents: &hash_table_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let token_bytes = (n as u64) * (std::mem::size_of::<Token>() as u64);
        let tokens_buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpzip-lz77-hash-tokens"),
            size: token_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpzip-lz77-hash-staging"),
            size: token_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = Params {
            input_len: n,
            hash_bits: HASH_BITS,
            window,
            min_match: MIN_MATCH,
            max_match: MAX_MATCH,
            _pad: [0; 3],
        };
        let params_buffer = self
            .ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpzip-lz77-hash-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let build_bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpzip-lz77-hash-build-bg"),
                layout: &self.build_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: hash_table.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });
        let lookup_bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpzip-lz77-hash-lookup-bg"),
                layout: &self.lookup_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: hash_table.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: tokens_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpzip-lz77-hash-enc"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpzip-lz77-hash-build-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.build_pipeline);
            pass.set_bind_group(0, &build_bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(64), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpzip-lz77-hash-lookup-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.lookup_pipeline);
            pass.set_bind_group(0, &lookup_bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&tokens_buffer, 0, &staging_buffer, 0, token_bytes);
        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("buffer map failed");

        let view = slice.get_mapped_range();
        let tokens: Vec<Token> = bytemuck::cast_slice::<u8, Token>(&view).to_vec();
        drop(view);
        staging_buffer.unmap();

        tokens
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

#[cfg(test)]
mod tests {
    use super::super::lz77::{greedy_walk, reconstruct};
    use super::*;

    fn try_pipeline() -> Option<Lz77HashPipeline> {
        let ctx = GpuContext::try_init().ok()?;
        Some(Lz77HashPipeline::new(Arc::new(ctx)))
    }

    fn round_trip(p: &Lz77HashPipeline, input: &[u8]) {
        let raw = p.match_find(input, DEFAULT_WINDOW);
        let walked = greedy_walk(&raw, input);
        let restored = reconstruct(&walked);
        assert_eq!(
            restored,
            input,
            "round trip mismatch on input len {}",
            input.len()
        );
    }

    #[test]
    fn empty_input() {
        let Some(p) = try_pipeline() else { return };
        assert!(p.match_find(&[], DEFAULT_WINDOW).is_empty());
    }

    #[test]
    fn small_no_match() {
        let Some(p) = try_pipeline() else { return };
        round_trip(&p, &(0..32u8).collect::<Vec<_>>());
    }

    #[test]
    fn obvious_repeat() {
        let Some(p) = try_pipeline() else { return };
        let input: Vec<u8> = b"abcabcabcabcabcabcabcabc".repeat(8).to_vec();
        round_trip(&p, &input);
    }

    #[test]
    fn self_overlap() {
        let Some(p) = try_pipeline() else { return };
        round_trip(&p, &[b'q'; 64]);
    }

    #[test]
    fn pseudo_random() {
        let Some(p) = try_pipeline() else { return };
        let input: Vec<u8> = (0..2048u32)
            .map(|i| (i.wrapping_mul(2654435761)) as u8)
            .collect();
        round_trip(&p, &input);
    }

    #[test]
    fn larger_text() {
        let Some(p) = try_pipeline() else { return };
        let mut input = Vec::new();
        for i in 0..512 {
            input.extend_from_slice(format!("line {i}: the quick brown fox\n").as_bytes());
        }
        round_trip(&p, &input);
    }
}
