//! Identity compute pipeline. Bytes go up to the GPU and come back
//! unchanged — bring-up scaffolding for the real LZ77 / Huffman shaders.

use std::sync::Arc;

use wgpu::util::DeviceExt;

use super::context::GpuContext;

const SHADER_SRC: &str = include_str!("identity.wgsl");

pub struct IdentityPipeline {
    ctx: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl IdentityPipeline {
    pub fn new(ctx: Arc<GpuContext>) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpzip-identity-shader"),
                source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
            });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("gpzip-identity-bgl"),
                    entries: &[
                        // Input buffer (read-only storage)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Output buffer (read_write storage)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpzip-identity-pl"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gpzip-identity-pipeline"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: "main",
            });

        Self {
            ctx,
            pipeline,
            bind_group_layout,
        }
    }

    /// Send `input` through the GPU. Returns the same bytes (copied via a
    /// compute shader). Lengths up to ~32 MiB are safe under default wgpu
    /// limits; larger inputs need the chunk pipeline above this layer.
    pub fn apply(&self, input: &[u8]) -> Vec<u8> {
        // u32-aligned padding: WGSL storage buffers are typed `array<u32>`,
        // so we round the byte buffer up to the next u32 boundary.
        let padded_len = input.len().next_multiple_of(4);
        let buffer_size = padded_len as u64;

        // Upload input. STORAGE | COPY_DST so we can write_buffer into it.
        let input_buffer = self
            .ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpzip-identity-input"),
                contents: &pad_to_u32(input),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        // Output: STORAGE | COPY_SRC so we can compute into it then copy to staging.
        let output_buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpzip-identity-output"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Staging: MAP_READ | COPY_DST so we can read it back on the CPU.
        let staging_buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpzip-identity-staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpzip-identity-bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpzip-identity-enc"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpzip-identity-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // workgroup_size is 64 (u32 lanes); padded_len/4 is the u32 count.
            let u32_count = (padded_len / 4) as u32;
            let workgroups = u32_count.div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, buffer_size);
        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        // Map staging buffer and wait for the GPU.
        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("map_async callback dropped without firing")
            .expect("buffer map failed");

        let view = slice.get_mapped_range();
        let mut out = view.to_vec();
        drop(view);
        staging_buffer.unmap();

        // Trim back down to the original (unpadded) length.
        out.truncate(input.len());
        out
    }
}

/// Right-pad with zeros to a u32 boundary. Necessary because the WGSL side
/// types the buffer as `array<u32>`.
fn pad_to_u32(input: &[u8]) -> Vec<u8> {
    let pad = input.len().next_multiple_of(4);
    if pad == input.len() {
        input.to_vec()
    } else {
        let mut v = Vec::with_capacity(pad);
        v.extend_from_slice(input);
        v.resize(pad, 0);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_pipeline() -> Option<IdentityPipeline> {
        let ctx = GpuContext::try_init().ok()?;
        Some(IdentityPipeline::new(Arc::new(ctx)))
    }

    #[test]
    fn identity_round_trips_short_buffer() {
        let Some(pipeline) = try_pipeline() else {
            eprintln!("skipping: no GPU adapter on this host");
            return;
        };
        let input: Vec<u8> = b"the quick brown fox jumps over the lazy dog".to_vec();
        let out = pipeline.apply(&input);
        assert_eq!(out, input);
    }

    #[test]
    fn identity_round_trips_unaligned_length() {
        let Some(pipeline) = try_pipeline() else {
            return;
        };
        let input: Vec<u8> = (0..1023u16).map(|i| (i % 251) as u8).collect();
        let out = pipeline.apply(&input);
        assert_eq!(out, input);
    }

    #[test]
    fn identity_round_trips_chunk_size() {
        let Some(pipeline) = try_pipeline() else {
            return;
        };
        let input: Vec<u8> = (0..(1 << 18)).map(|i: u32| (i % 251) as u8).collect();
        let out = pipeline.apply(&input);
        assert_eq!(out, input);
    }
}
