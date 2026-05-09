//! D-2: full-GPU fixed-Huffman DEFLATE emission.
//!
//! Where D-1 had the host build a pre-laid-out atom list and the GPU
//! only do the parallel `atomicOr` placement, D-2 keeps everything
//! GPU-side after the token upload:
//!
//! 1. **compute_and_local_scan** — per-token compute of bit length
//!    using uploaded Huffman / length-symbol / distance-symbol LUTs,
//!    then a workgroup-local Hillis–Steele scan into per-token
//!    exclusive-prefix offsets. Each workgroup also writes its
//!    inclusive total to `workgroup_totals[wg_id]`.
//!
//! 2. **scan_totals** — single-workgroup scan of the (≤ WG_SIZE)
//!    workgroup_totals into `workgroup_bases[wg_id]` (exclusive
//!    prefix), bridging the per-workgroup scans into a global one.
//!
//! 3. **emit** — per-token thread reads its global bit offset
//!    (= `workgroup_bases[wid] + per_token_offset[i] + 3` for the
//!    block header) and `atomicOr`s its 1–4 emissions. The leading
//!    thread of workgroup 0 emits the block header; the first thread
//!    past the last token emits EOB.
//!
//! Status: PoC, fixed-Huffman only. Production swap requires D-3
//! (dynamic Huffman header on GPU). Output bytes are bit-identical to
//! `deflate::encode_fixed_block` and the D-1 emit shader on the same
//! token stream — verified by round-trip and direct `assert_eq!`
//! against D-1 output in the bench.

#![allow(dead_code)]

use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::context::GpuContext;
use super::lz77::Token;

/// Maximum token count the pool's buffers handle in one call. Chunks
/// larger than this would need a fresh, bigger BufferSet (we'd panic
/// before then because n_workgroups > WG_SIZE breaks the single-pass
/// scan_totals; see assert in dispatch_emit). 32 KiB matches
/// `GpuBackend::chunk_size`.
const MAX_TOKENS: usize = 32 * 1024;
/// Max output bytes for the worst case: ceil((header + N*32 + 16) / 8) + slack.
/// Header is at most ~512 bytes for an extreme dynamic block; round generously.
const MAX_OUTPUT_BYTES: usize = MAX_TOKENS * 4 + 1024;
/// Max bytes for the host-built block header (dynamic Huffman headers
/// max out around 300 bytes in practice; 1 KiB is safe slack).
const MAX_HEADER_BYTES: usize = 1024;

const SHADER: &str = include_str!("huffman_emit_v2.wgsl");
const WG_SIZE: u32 = 256;

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Params {
    n_tokens: u32,
    n_workgroups: u32,
    /// Bit count of the host-pre-written block header at output[0..]; the
    /// emit shader places token bits starting here. For fixed Huffman
    /// the header is just BFINAL=1 + BTYPE=01 = 3 bits; for dynamic
    /// (D-3) it's the full RFC 1951 §3.2.7 header.
    header_bit_count: u32,
    _pad: u32,
}

pub struct HuffmanEmitV2Pipeline {
    ctx: Arc<GpuContext>,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_compute: wgpu::ComputePipeline,
    pipeline_scan: wgpu::ComputePipeline,
    pipeline_emit: wgpu::ComputePipeline,
    // Pre-uploaded fixed-Huffman LUT buffers — built once at pipeline
    // creation and reused across every emit call. The fixed-block path
    // binds these directly; the dynamic-block path binds the per-call
    // Huffman buffers from `EmitBufferSet` instead.
    buf_lit_lens: wgpu::Buffer,
    buf_lit_codes_pre: wgpu::Buffer,
    buf_dist_lens: wgpu::Buffer,
    buf_dist_codes_pre: wgpu::Buffer,
    buf_len_lut: wgpu::Buffer,
    buf_dist_lut_lo: wgpu::Buffer,
    buf_dist_lut_hi: wgpu::Buffer,
    /// Pool of variable-data buffers reused across calls. Each set is
    /// sized for the maximum supported chunk (32 KiB tokens). D-1 / D-2
    /// allocated these per call; under concurrent load the wgpu /
    /// driver bookkeeping for ~10 buffer creations dominated short
    /// dispatches, especially noticeable on the dynamic path which adds
    /// 4 more per-call Huffman buffers.
    pool: Mutex<Vec<EmitBufferSet>>,
}

/// One reusable bundle of GPU buffers for an emit call. All sized for
/// the worst case (32 KiB chunk → 32 768 tokens). Smaller chunks just
/// leave the buffer tails unused; the params + dispatch counts cap how
/// far the shader reads / writes.
struct EmitBufferSet {
    /// Packed tokens (u32 each).
    tokens: wgpu::Buffer,
    /// Per-call Huffman tables (only written by the dynamic path).
    /// Sized for 288 / 30 entries respectively.
    lit_lens: wgpu::Buffer,
    lit_codes_pre: wgpu::Buffer,
    dist_lens: wgpu::Buffer,
    dist_codes_pre: wgpu::Buffer,
    /// Block header bytes the host pre-builds; copied into output.
    header: wgpu::Buffer,
    /// Per-token exclusive prefix offset within its workgroup
    /// (output of pass 1).
    per_token_offset: wgpu::Buffer,
    /// Per-workgroup totals (output of pass 1, input of pass 2).
    workgroup_totals: wgpu::Buffer,
    /// Per-workgroup base offsets (exclusive prefix of totals,
    /// output of pass 2).
    workgroup_bases: wgpu::Buffer,
    /// Atomic-OR target for the bitstream.
    output: wgpu::Buffer,
    /// MAP_READ buffer for `output` readback.
    staging: wgpu::Buffer,
    /// Uniform with n_tokens / n_workgroups / header_bit_count.
    params: wgpu::Buffer,
}

impl HuffmanEmitV2Pipeline {
    pub fn new(ctx: Arc<GpuContext>) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpzip-huffman-emit-v2-shader"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });

        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpzip-huffman-emit-v2-bgl"),
                entries: &[
                    storage_entry(0, true),   // tokens
                    storage_entry(1, true),   // lit_lens
                    storage_entry(2, true),   // lit_codes_pre
                    storage_entry(3, true),   // dist_lens
                    storage_entry(4, true),   // dist_codes_pre
                    storage_entry(5, true),   // len_lut
                    storage_entry(6, true),   // dist_lut_lo
                    storage_entry(7, true),   // dist_lut_hi
                    storage_entry(8, false),  // per_token_offset
                    storage_entry(9, false),  // workgroup_totals
                    storage_entry(10, false), // workgroup_bases
                    storage_entry(11, false), // output (atomic)
                    uniform_entry(12),
                ],
            });
        let pl_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpzip-huffman-emit-v2-pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
        let make_pipeline = |entry: &str, label: &str| {
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pl_layout),
                    module: &module,
                    entry_point: entry,
                })
        };
        let pipeline_compute = make_pipeline("compute_and_local_scan", "v2-compute-scan");
        let pipeline_scan = make_pipeline("scan_totals", "v2-scan-totals");
        let pipeline_emit = make_pipeline("emit", "v2-emit");

        // Upload the Huffman/symbol LUTs once. They never change for
        // fixed Huffman.
        let lit_lens = build_fixed_lit_lens();
        let lit_codes_pre = build_fixed_lit_codes_pre();
        let dist_lens = build_fixed_dist_lens();
        let dist_codes_pre = build_fixed_dist_codes_pre();
        let len_lut = build_len_lut();
        let dist_lut_lo = build_dist_lut_lo();
        let dist_lut_hi = build_dist_lut_hi();

        let mk_storage = |name: &str, data: &[u32]| {
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(name),
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        Self {
            buf_lit_lens: mk_storage("v2-lit-lens", &lit_lens),
            buf_lit_codes_pre: mk_storage("v2-lit-codes-pre", &lit_codes_pre),
            buf_dist_lens: mk_storage("v2-dist-lens", &dist_lens),
            buf_dist_codes_pre: mk_storage("v2-dist-codes-pre", &dist_codes_pre),
            buf_len_lut: mk_storage("v2-len-lut", &len_lut),
            buf_dist_lut_lo: mk_storage("v2-dist-lut-lo", &dist_lut_lo),
            buf_dist_lut_hi: mk_storage("v2-dist-lut-hi", &dist_lut_hi),
            ctx,
            bind_group_layout: bgl,
            pipeline_compute,
            pipeline_scan,
            pipeline_emit,
            pool: Mutex::new(Vec::new()),
        }
    }

    /// Pop a buffer set from the pool, or build a fresh one if empty.
    fn acquire(&self) -> EmitBufferSet {
        if let Ok(mut p) = self.pool.lock() {
            if let Some(s) = p.pop() {
                return s;
            }
        }
        EmitBufferSet::new(&self.ctx)
    }

    /// Return a buffer set to the pool. Capped at 8 sets so concurrent
    /// dispatches don't pile up unbounded GPU memory.
    fn release(&self, set: EmitBufferSet) {
        if let Ok(mut p) = self.pool.lock() {
            if p.len() < 8 {
                p.push(set);
            }
        }
    }

    fn upload_u32(&self, label: &str, data: &[u32]) -> wgpu::Buffer {
        self.ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    /// Encode `tokens` as a fixed-Huffman DEFLATE block on the GPU,
    /// computing bit lengths and bit offsets entirely on the GPU. Output
    /// is bit-identical to `encode_fixed_block` (host) and to D-1's
    /// `emit_fixed_block`.
    pub fn emit_fixed_block_v2(&self, tokens: &[Token]) -> Vec<u8> {
        // Empty token stream: just header + EOB. Build directly on host
        // — not worth a GPU dispatch for 12 bits.
        if tokens.is_empty() {
            return empty_fixed_block();
        }

        // Fixed-Huffman block header is just BFINAL=1 + BTYPE=01 = 0b011
        // packed LSB-first, i.e. byte 0x03 (3 bits).
        let header_bytes = vec![0x03u8];
        let header_bit_count: u32 = 3;
        let total_bits = total_bits_for_fixed(tokens);

        self.dispatch_emit(
            tokens,
            HuffmanSource::Static,
            &header_bytes,
            header_bit_count,
            total_bits,
        )
    }

    /// D-3: dynamic-Huffman DEFLATE block, GPU-emitted. The host builds
    /// the per-block Huffman trees + RFC 1951 §3.2.7 header bitstream
    /// (same logic as `deflate::try_write_dynamic_block`'s header half),
    /// then uploads the resulting code-length / pre-reversed-code arrays
    /// and the header bytes; the GPU dispatch is identical to the fixed
    /// path, just with different uploaded tables and a longer header.
    ///
    /// Returns `Err` if the Huffman tree exceeds DEFLATE's 15-bit cap
    /// (extremely unusual; the caller can fall back to
    /// `emit_fixed_block_v2`).
    pub fn emit_dynamic_block_v3(&self, tokens: &[Token]) -> std::io::Result<Vec<u8>> {
        if tokens.is_empty() {
            return Ok(empty_fixed_block());
        }
        let dyn_h = build_dynamic_huffman(tokens)
            .ok_or_else(|| std::io::Error::other("dynamic Huffman build failed (tree too tall)"))?;
        let total_bits = dyn_h.header_bit_count as u64 + dyn_h.body_bit_count + dyn_h.eob_bits;
        Ok(self.dispatch_emit(
            tokens,
            HuffmanSource::Dynamic(&dyn_h),
            &dyn_h.header_bytes,
            dyn_h.header_bit_count,
            total_bits,
        ))
    }

    /// Shared 3-pass dispatch used by both fixed and dynamic emission.
    /// Pulls a buffer set from the pool, writes the per-call data into
    /// it, dispatches the three passes, reads back, returns the set to
    /// the pool. Avoids the wgpu/driver overhead of creating ~10 fresh
    /// buffers per call (the dominant cost on D-3 single-call benches).
    fn dispatch_emit(
        &self,
        tokens: &[Token],
        huffman: HuffmanSource<'_>,
        header_bytes: &[u8],
        header_bit_count: u32,
        total_bits: u64,
    ) -> Vec<u8> {
        let packed: Vec<u32> = tokens
            .iter()
            .map(|t| (t.length << 16) | (t.distance & 0xFFFF))
            .collect();
        let n_tokens = packed.len() as u32;
        let n_workgroups = n_tokens.div_ceil(WG_SIZE);
        assert!(
            n_workgroups <= WG_SIZE,
            "v2/v3 single-pass scan_totals only handles ≤ WG_SIZE workgroups; \
             n_tokens={n_tokens} requires {n_workgroups} (max {WG_SIZE})"
        );

        // Header padded to a 4-byte multiple (wgpu copy alignment).
        let mut header_padded = header_bytes.to_vec();
        while header_padded.len() % 4 != 0 {
            header_padded.push(0);
        }

        // Acquire a pooled buffer set; build a fresh one if the pool's
        // empty.
        let set = self.acquire();

        // Per-call data: write into the pooled buffers via queue. wgpu
        // schedules these to land before the encoder runs at submit time.
        self.ctx
            .queue
            .write_buffer(&set.tokens, 0, bytemuck::cast_slice(&packed));
        self.ctx.queue.write_buffer(&set.header, 0, &header_padded);

        // Dynamic Huffman: write per-block code length / pre-reversed
        // code arrays into the pooled buffers. Static path leaves them
        // untouched (we'll bind the pipeline's pre-uploaded fixed LUTs
        // in the bind group).
        let (lit_lens_buf, lit_codes_pre_buf, dist_lens_buf, dist_codes_pre_buf) = match huffman {
            HuffmanSource::Static => (
                &self.buf_lit_lens,
                &self.buf_lit_codes_pre,
                &self.buf_dist_lens,
                &self.buf_dist_codes_pre,
            ),
            HuffmanSource::Dynamic(h) => {
                self.ctx.queue.write_buffer(
                    &set.lit_lens,
                    0,
                    bytemuck::cast_slice(&h.lit_lens_u32),
                );
                self.ctx.queue.write_buffer(
                    &set.lit_codes_pre,
                    0,
                    bytemuck::cast_slice(&h.lit_codes_pre),
                );
                self.ctx.queue.write_buffer(
                    &set.dist_lens,
                    0,
                    bytemuck::cast_slice(&h.dist_lens_u32),
                );
                self.ctx.queue.write_buffer(
                    &set.dist_codes_pre,
                    0,
                    bytemuck::cast_slice(&h.dist_codes_pre),
                );
                (
                    &set.lit_lens,
                    &set.lit_codes_pre,
                    &set.dist_lens,
                    &set.dist_codes_pre,
                )
            }
        };

        let params = Params {
            n_tokens,
            n_workgroups,
            header_bit_count,
            _pad: 0,
        };
        self.ctx
            .queue
            .write_buffer(&set.params, 0, bytemuck::bytes_of(&params));

        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("v2-bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    bge(0, &set.tokens),
                    bge(1, lit_lens_buf),
                    bge(2, lit_codes_pre_buf),
                    bge(3, dist_lens_buf),
                    bge(4, dist_codes_pre_buf),
                    bge(5, &self.buf_len_lut),
                    bge(6, &self.buf_dist_lut_lo),
                    bge(7, &self.buf_dist_lut_hi),
                    bge(8, &set.per_token_offset),
                    bge(9, &set.workgroup_totals),
                    bge(10, &set.workgroup_bases),
                    bge(11, &set.output),
                    bge(12, &set.params),
                ],
            });

        // We only ever fill the first `live_bytes` of output / staging
        // with meaningful data; clear/copy/copy-back at that size to
        // skip touching the unused tail of the MAX_OUTPUT_BYTES buffer.
        let live_bits = (header_bit_count as u64) + (n_tokens as u64) * 32 + 16;
        let live_words = (live_bits as usize).div_ceil(32) + 1;
        let live_bytes = (live_words * 4) as u64;

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("v2-enc"),
            });
        // 1. Zero the live region of the output buffer.
        encoder.clear_buffer(&set.output, 0, Some(live_bytes));
        // 2. Copy the host-built block header into output[0..].
        encoder.copy_buffer_to_buffer(&set.header, 0, &set.output, 0, header_padded.len() as u64);
        // 3. compute + local scan
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("v2-compute-scan-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_compute);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n_workgroups, 1, 1);
        }
        // 4. scan workgroup totals (single workgroup)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("v2-scan-totals-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_scan);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        // 5. emit (one extra thread for EOB)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("v2-emit-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_emit);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((n_tokens + 1).div_ceil(WG_SIZE), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&set.output, 0, &set.staging, 0, live_bytes);
        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        let slice = set.staging.slice(0..live_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("v2 buffer map failed");
        let view = slice.get_mapped_range();
        let n_bytes = (total_bits as usize).div_ceil(8);
        let out = view[..n_bytes].to_vec();
        drop(view);
        set.staging.unmap();

        self.release(set);
        out
    }
}

/// Source of the four per-block Huffman lookup buffers (literal/length
/// codes + lengths, distance codes + lengths). Fixed Huffman uses the
/// pipeline's pre-uploaded buffers; dynamic computes them per block.
enum HuffmanSource<'a> {
    Static,
    Dynamic(&'a DynamicHuffman),
}

impl EmitBufferSet {
    fn new(ctx: &GpuContext) -> Self {
        let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        // STORAGE | COPY_DST so queue.write_buffer can populate them.
        let storage_dst = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        Self {
            tokens: mk("v2-pool-tokens", (MAX_TOKENS * 4) as u64, storage_dst),
            lit_lens: mk("v2-pool-lit-lens", 288 * 4, storage_dst),
            lit_codes_pre: mk("v2-pool-lit-codes-pre", 288 * 4, storage_dst),
            dist_lens: mk("v2-pool-dist-lens", 32 * 4, storage_dst),
            dist_codes_pre: mk("v2-pool-dist-codes-pre", 32 * 4, storage_dst),
            header: mk(
                "v2-pool-header",
                MAX_HEADER_BYTES as u64,
                wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            ),
            per_token_offset: mk(
                "v2-pool-per-token-offset",
                (MAX_TOKENS * 4) as u64,
                wgpu::BufferUsages::STORAGE,
            ),
            workgroup_totals: mk(
                "v2-pool-workgroup-totals",
                (WG_SIZE as u64) * 4,
                wgpu::BufferUsages::STORAGE,
            ),
            workgroup_bases: mk(
                "v2-pool-workgroup-bases",
                (WG_SIZE as u64) * 4,
                wgpu::BufferUsages::STORAGE,
            ),
            output: mk(
                "v2-pool-output",
                MAX_OUTPUT_BYTES as u64,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            ),
            staging: mk(
                "v2-pool-staging",
                MAX_OUTPUT_BYTES as u64,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
            params: mk(
                "v2-pool-params",
                std::mem::size_of::<Params>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
        }
    }
}

// ============================================================
// LUT builders + host-side fixed Huffman tables
// ============================================================

fn fixed_litlen(symbol: u32) -> (u32, u32) {
    match symbol {
        0..=143 => (0b0011_0000 + symbol, 8),
        144..=255 => (0b1_1001_0000 + (symbol - 144), 9),
        256..=279 => (symbol - 256, 7),
        280..=287 => (0b1100_0000 + (symbol - 280), 8),
        _ => unreachable!(),
    }
}

fn fixed_dist(symbol: u32) -> (u32, u32) {
    (symbol, 5)
}

#[inline]
fn rev_bits(code: u32, bits: u32) -> u32 {
    if bits == 0 {
        0
    } else {
        code.reverse_bits() >> (32 - bits)
    }
}

fn build_fixed_lit_lens() -> Vec<u32> {
    (0..288u32).map(|s| fixed_litlen(s).1).collect()
}
fn build_fixed_lit_codes_pre() -> Vec<u32> {
    (0..288u32)
        .map(|s| {
            let (c, b) = fixed_litlen(s);
            rev_bits(c, b)
        })
        .collect()
}
fn build_fixed_dist_lens() -> Vec<u32> {
    (0..30u32).map(|s| fixed_dist(s).1).collect()
}
fn build_fixed_dist_codes_pre() -> Vec<u32> {
    (0..30u32)
        .map(|s| {
            let (c, b) = fixed_dist(s);
            rev_bits(c, b)
        })
        .collect()
}

/// Length 3..=258 → packed (sym-257) | (extra<<8) | (base<<16). Length
/// indices 0..3 are unused (length must be ≥ 3) and stay zero.
fn build_len_lut() -> Vec<u32> {
    const ROWS: &[(u32, u32, u32)] = &[
        (3, 0, 257),
        (4, 0, 258),
        (5, 0, 259),
        (6, 0, 260),
        (7, 0, 261),
        (8, 0, 262),
        (9, 0, 263),
        (10, 0, 264),
        (11, 1, 265),
        (13, 1, 266),
        (15, 1, 267),
        (17, 1, 268),
        (19, 2, 269),
        (23, 2, 270),
        (27, 2, 271),
        (31, 2, 272),
        (35, 3, 273),
        (43, 3, 274),
        (51, 3, 275),
        (59, 3, 276),
        (67, 4, 277),
        (83, 4, 278),
        (99, 4, 279),
        (115, 4, 280),
        (131, 5, 281),
        (163, 5, 282),
        (195, 5, 283),
        (227, 5, 284),
        (258, 0, 285),
    ];
    let mut lut = vec![0u32; 259];
    for (idx, &(base, extra, sym)) in ROWS.iter().enumerate() {
        let next = if idx + 1 < ROWS.len() {
            ROWS[idx + 1].0
        } else {
            259
        };
        for len in base..next {
            lut[len as usize] = (sym - 257) | (extra << 8) | (base << 16);
        }
    }
    lut
}

const DIST_ROWS: &[(u32, u32, u32)] = &[
    (1, 0, 0),
    (2, 0, 1),
    (3, 0, 2),
    (4, 0, 3),
    (5, 1, 4),
    (7, 1, 5),
    (9, 2, 6),
    (13, 2, 7),
    (17, 3, 8),
    (25, 3, 9),
    (33, 4, 10),
    (49, 4, 11),
    (65, 5, 12),
    (97, 5, 13),
    (129, 6, 14),
    (193, 6, 15),
    (257, 7, 16),
    (385, 7, 17),
    (513, 8, 18),
    (769, 8, 19),
    (1025, 9, 20),
    (1537, 9, 21),
    (2049, 10, 22),
    (3073, 10, 23),
    (4097, 11, 24),
    (6145, 11, 25),
    (8193, 12, 26),
    (12289, 12, 27),
    (16385, 13, 28),
    (24577, 13, 29),
];

/// Distance 1..=256 → packed sym | (extra<<8) | (base<<16). Index 0 is
/// unused (distance must be ≥ 1).
fn build_dist_lut_lo() -> Vec<u32> {
    let mut lut = vec![0u32; 257];
    for (idx, &(base, extra, sym)) in DIST_ROWS.iter().enumerate() {
        if base > 256 {
            break;
        }
        let next = if idx + 1 < DIST_ROWS.len() {
            DIST_ROWS[idx + 1].0
        } else {
            32769
        };
        let end = next.min(257);
        for d in base..end {
            lut[d as usize] = sym | (extra << 8) | (base << 16);
        }
    }
    lut
}

/// Distance > 256: indexed by (d-1)>>7, gives the same packed
/// representation. Index space is 256 entries (covers up to d = 32768).
fn build_dist_lut_hi() -> Vec<u32> {
    let mut lut = vec![0u32; 256];
    for (idx, &(base, extra, sym)) in DIST_ROWS.iter().enumerate() {
        if base < 257 {
            continue;
        }
        let next = if idx + 1 < DIST_ROWS.len() {
            DIST_ROWS[idx + 1].0
        } else {
            32769
        };
        for d in base..next {
            let h_idx = ((d - 1) >> 7) as usize;
            if h_idx < lut.len() {
                lut[h_idx] = sym | (extra << 8) | (base << 16);
            }
        }
    }
    lut
}

/// Compute total DEFLATE bits for a fixed-Huffman block over `tokens`.
/// Mirrors the GPU's per-token compute + the host's header/EOB framing.
fn total_bits_for_fixed(tokens: &[Token]) -> u64 {
    let mut bits: u64 = 3; // BFINAL + BTYPE
    for t in tokens {
        if t.is_literal() {
            let (_, b) = fixed_litlen(t.distance);
            bits += b as u64;
        } else {
            let (lcode, lextra, _lval) = host_length_code(t.length);
            let (_, lb) = fixed_litlen(lcode);
            bits += (lb + lextra) as u64;
            let (_dcode, dextra, _dval) = host_distance_code(t.distance);
            bits += (5 + dextra) as u64;
        }
    }
    let (_, eob_bits) = fixed_litlen(256);
    bits + eob_bits as u64
}

/// Host-side O(1) lookup for length/distance code metadata. The data
/// is the same as the GPU-side `len_lut()` / `dist_lut_lo()` /
/// `dist_lut_hi()` (packed `(sym - 257) | (extra << 8) | (base << 16)`
/// for length, `sym | (extra << 8) | (base << 16)` for distance), built
/// once on first call and cached. Replaces the original 29-row linear
/// scans which were called O(n_tokens) times per dynamic encode block.
fn host_len_lut() -> &'static [u32; 259] {
    use std::sync::OnceLock;
    static LUT: OnceLock<[u32; 259]> = OnceLock::new();
    LUT.get_or_init(|| {
        let v = build_len_lut();
        let mut a = [0u32; 259];
        a[..v.len()].copy_from_slice(&v);
        a
    })
}

fn host_dist_lut_lo() -> &'static [u32; 257] {
    use std::sync::OnceLock;
    static LUT: OnceLock<[u32; 257]> = OnceLock::new();
    LUT.get_or_init(|| {
        let v = build_dist_lut_lo();
        let mut a = [0u32; 257];
        a[..v.len()].copy_from_slice(&v);
        a
    })
}

fn host_dist_lut_hi() -> &'static [u32; 256] {
    use std::sync::OnceLock;
    static LUT: OnceLock<[u32; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let v = build_dist_lut_hi();
        let mut a = [0u32; 256];
        a[..v.len()].copy_from_slice(&v);
        a
    })
}

fn host_length_code(length: u32) -> (u32, u32, u32) {
    debug_assert!((3..=258).contains(&length));
    let packed = host_len_lut()[length as usize];
    let sym = (packed & 0xFF) + 257;
    let extra = (packed >> 8) & 0xFF;
    let base = (packed >> 16) & 0xFFFF;
    (sym, extra, length - base)
}

fn host_distance_code(distance: u32) -> (u32, u32, u32) {
    debug_assert!((1..=32768).contains(&distance));
    let packed = if distance <= 256 {
        host_dist_lut_lo()[distance as usize]
    } else {
        host_dist_lut_hi()[((distance - 1) >> 7) as usize]
    };
    let sym = packed & 0xFF;
    let extra = (packed >> 8) & 0xFF;
    let base = (packed >> 16) & 0xFFFF;
    (sym, extra, distance - base)
}

const LEN_ROWS: &[(u32, u32, u32)] = &[
    (3, 0, 257),
    (4, 0, 258),
    (5, 0, 259),
    (6, 0, 260),
    (7, 0, 261),
    (8, 0, 262),
    (9, 0, 263),
    (10, 0, 264),
    (11, 1, 265),
    (13, 1, 266),
    (15, 1, 267),
    (17, 1, 268),
    (19, 2, 269),
    (23, 2, 270),
    (27, 2, 271),
    (31, 2, 272),
    (35, 3, 273),
    (43, 3, 274),
    (51, 3, 275),
    (59, 3, 276),
    (67, 4, 277),
    (83, 4, 278),
    (99, 4, 279),
    (115, 4, 280),
    (131, 5, 281),
    (163, 5, 282),
    (195, 5, 283),
    (227, 5, 284),
    (258, 0, 285),
];

/// Fixed-Huffman empty block: header (BFINAL=1, BTYPE=01, 0b011 LSB)
/// then EOB (symbol 256, fixed code = 7-bit `0000000` = 0). Total 10
/// bits → 2 bytes. Used as a fast path when there are no tokens.
fn empty_fixed_block() -> Vec<u8> {
    let mut out = vec![0u8; 2];
    // Bit 0..2: 0b011
    out[0] = 0b011;
    // EOB code is 0 in 7 bits, written LSB-first → adds 7 zero bits.
    // Already zero; nothing to do.
    out
}

// ============================================================
// Dynamic Huffman header builder (D-3)
// ============================================================

use super::huffman::{canonical_codes, try_build_code_lengths};

/// RFC 1951 §3.2.7: order in which the meta-Huffman code lengths are
/// written.
const META_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// Per-block state needed to dispatch the GPU emit shaders for a
/// dynamic Huffman block: the per-symbol code lengths and pre-reversed
/// codes that the shader looks up, plus the host-built header bitstream
/// the shader writes verbatim before the per-token bits.
pub(crate) struct DynamicHuffman {
    pub lit_lens_u32: Vec<u32>,  // 286 → padded to 288 for alignment
    pub lit_codes_pre: Vec<u32>, // pre-reversed canonical codes
    pub dist_lens_u32: Vec<u32>, // 30 → padded to 32
    pub dist_codes_pre: Vec<u32>,
    pub header_bytes: Vec<u8>,
    pub header_bit_count: u32,
    /// Total per-token bits (sum of bit lengths emitted by the per-token
    /// shader). Computed during the build pass alongside the frequencies.
    pub body_bit_count: u64,
    /// Bit count of the EOB code (the shader emits it after the body).
    pub eob_bits: u64,
}

/// Build a dynamic-Huffman block's header + per-symbol lookup tables for
/// `tokens`. Mirrors the header half of `deflate::try_write_dynamic_block`.
/// Returns `None` if the natural Huffman tree exceeds the 15-bit DEFLATE
/// cap (caller falls back to fixed encoding).
pub(crate) fn build_dynamic_huffman(tokens: &[Token]) -> Option<DynamicHuffman> {
    // 1. Frequencies. Always count EOB at least once. Bit count of the
    // body is computed in step 4 below, after we have the code lengths.
    let mut litlen_freq = [0u32; 286];
    let mut dist_freq = [0u32; 30];
    for tok in tokens {
        if tok.is_literal() {
            litlen_freq[tok.distance as usize] += 1;
        } else {
            let (lcode, _, _) = host_length_code(tok.length);
            litlen_freq[lcode as usize] += 1;
            let (dcode, _, _) = host_distance_code(tok.distance);
            dist_freq[dcode as usize] += 1;
        }
    }
    litlen_freq[256] += 1;

    // 2. Length-limited Huffman trees (15-bit cap on litlen+dist, 7-bit
    // on the meta-Huffman). If any fail, signal fallback.
    let mut litlen_lens = try_build_code_lengths(&litlen_freq, 15)?;
    let mut dist_lens = try_build_code_lengths(&dist_freq, 15)?;

    // DEFLATE quirk: the distance Huffman tree must have at least two
    // leaves (or be empty entirely with HDIST=0). Match the existing
    // host fixup so we produce a decodable block in pathological
    // single-distance cases.
    let used_dist: usize = dist_lens.iter().filter(|&&l| l > 0).count();
    if used_dist == 0 {
        dist_lens[0] = 1;
        dist_lens[1] = 1;
    } else if used_dist == 1 {
        let only = dist_lens.iter().position(|&l| l > 0).unwrap();
        dist_lens[only] = 1;
        dist_lens[1 - only.min(1)] = 1;
    }
    if litlen_lens.iter().filter(|&&l| l > 0).count() < 2 {
        litlen_lens[0] = 1;
        litlen_lens[256] = 1;
    }

    let litlen_codes = canonical_codes(&litlen_lens);
    let dist_codes = canonical_codes(&dist_lens);

    // 3. Pre-reverse codes for LSB-first packing.
    let lit_codes_pre: Vec<u32> = litlen_codes
        .iter()
        .zip(&litlen_lens)
        .map(|(&c, &l)| if l == 0 { 0 } else { rev_bits(c, l as u32) })
        .collect();
    let dist_codes_pre: Vec<u32> = dist_codes
        .iter()
        .zip(&dist_lens)
        .map(|(&c, &l)| if l == 0 { 0 } else { rev_bits(c, l as u32) })
        .collect();

    // 4. Now that we have the code lengths, walk tokens once more to
    //    sum per-token bits.
    let mut body_bits: u64 = 0;
    for tok in tokens {
        if tok.is_literal() {
            body_bits += litlen_lens[tok.distance as usize] as u64;
        } else {
            let (lcode, lextra, _) = host_length_code(tok.length);
            body_bits += litlen_lens[lcode as usize] as u64 + lextra as u64;
            let (dcode, dextra, _) = host_distance_code(tok.distance);
            body_bits += dist_lens[dcode as usize] as u64 + dextra as u64;
        }
    }
    let eob_bits = litlen_lens[256] as u64;

    // 5. HLIT / HDIST = trim trailing zeros (keep min 257 / 1).
    let nlit = trim_to_min(&litlen_lens, 257);
    let ndist = trim_to_min(&dist_lens, 1);
    let hlit = (nlit - 257) as u32;
    let hdist = (ndist - 1) as u32;

    // 6. RLE-encode the combined code-length sequence.
    let mut combined = Vec::with_capacity(nlit + ndist);
    combined.extend_from_slice(&litlen_lens[..nlit]);
    combined.extend_from_slice(&dist_lens[..ndist]);
    let rle = rle_encode_lengths(&combined);

    // 7. Meta-Huffman from the RLE symbols (max 7-bit codes).
    let mut meta_freq = [0u32; 19];
    for e in &rle {
        meta_freq[e.code as usize] += 1;
    }
    let meta_lens = try_build_code_lengths(&meta_freq, 7)?;
    let meta_codes = canonical_codes(&meta_lens);

    // 8. HCLEN: number of meta-Huffman code lengths to write, in
    // META_ORDER. Trim trailing zeros but keep at least 4.
    let mut hclen_count = 19;
    while hclen_count > 4 && meta_lens[META_ORDER[hclen_count - 1]] == 0 {
        hclen_count -= 1;
    }
    let hclen = (hclen_count - 4) as u32;

    // 9. Write header bits: BFINAL=1, BTYPE=10 → 0b101, then HLIT (5),
    // HDIST (5), HCLEN (4), then meta lens (3 bits each in META_ORDER),
    // then the RLE-encoded code-length sequence (meta-Huffman emit +
    // extra bits per RLE entry).
    let mut bw = HostBitWriter::new();
    bw.write_bits(0b101, 3);
    bw.write_bits(hlit, 5);
    bw.write_bits(hdist, 5);
    bw.write_bits(hclen, 4);
    for i in 0..hclen_count {
        bw.write_bits(meta_lens[META_ORDER[i]] as u32, 3);
    }
    for e in &rle {
        let code = meta_codes[e.code as usize];
        let bits = meta_lens[e.code as usize] as u32;
        bw.write_huffman(code, bits);
        if e.extra_bits > 0 {
            bw.write_bits(e.extra_value, e.extra_bits);
        }
    }
    let header_bit_count = bw.bit_count();
    let header_bytes = bw.into_bytes();

    Some(DynamicHuffman {
        lit_lens_u32: litlen_lens.iter().map(|&b| b as u32).collect(),
        lit_codes_pre,
        dist_lens_u32: dist_lens.iter().map(|&b| b as u32).collect(),
        dist_codes_pre,
        header_bytes,
        header_bit_count,
        body_bit_count: body_bits,
        eob_bits,
    })
}

/// Single RLE entry for the code-length sequence (RFC 1951 §3.2.7).
/// Same shape as `deflate::RleEntry`.
struct RleEntry {
    code: u8,
    extra_bits: u32,
    extra_value: u32,
}

fn rle_encode_lengths(lens: &[u8]) -> Vec<RleEntry> {
    let mut out = Vec::with_capacity(lens.len());
    let mut i = 0;
    while i < lens.len() {
        let cur = lens[i];
        let mut run = 1usize;
        while i + run < lens.len() && lens[i + run] == cur {
            run += 1;
        }
        if cur == 0 {
            while run >= 11 {
                let n = run.min(138);
                out.push(RleEntry {
                    code: 18,
                    extra_bits: 7,
                    extra_value: (n - 11) as u32,
                });
                run -= n;
                i += n;
            }
            while run >= 3 {
                let n = run.min(10);
                out.push(RleEntry {
                    code: 17,
                    extra_bits: 3,
                    extra_value: (n - 3) as u32,
                });
                run -= n;
                i += n;
            }
            while run > 0 {
                out.push(RleEntry {
                    code: 0,
                    extra_bits: 0,
                    extra_value: 0,
                });
                run -= 1;
                i += 1;
            }
        } else {
            out.push(RleEntry {
                code: cur,
                extra_bits: 0,
                extra_value: 0,
            });
            i += 1;
            run -= 1;
            while run >= 3 {
                let n = run.min(6);
                out.push(RleEntry {
                    code: 16,
                    extra_bits: 2,
                    extra_value: (n - 3) as u32,
                });
                run -= n;
                i += n;
            }
            while run > 0 {
                out.push(RleEntry {
                    code: cur,
                    extra_bits: 0,
                    extra_value: 0,
                });
                run -= 1;
                i += 1;
            }
        }
    }
    out
}

fn trim_to_min(lens: &[u8], min: usize) -> usize {
    let mut n = lens.len();
    while n > min && lens[n - 1] == 0 {
        n -= 1;
    }
    n
}

/// Tiny DEFLATE-conventions bit writer used only to build the dynamic
/// header bitstream on the host. The token body is built on the GPU.
struct HostBitWriter {
    out: Vec<u8>,
    accum: u32,
    nbits: u32,
    total: u32,
}

impl HostBitWriter {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            accum: 0,
            nbits: 0,
            total: 0,
        }
    }
    fn write_bits(&mut self, value: u32, count: u32) {
        debug_assert!(count <= 24);
        self.accum |= (value & ((1u32 << count) - 1)) << self.nbits;
        self.nbits += count;
        self.total += count;
        while self.nbits >= 8 {
            self.out.push((self.accum & 0xff) as u8);
            self.accum >>= 8;
            self.nbits -= 8;
        }
    }
    fn write_huffman(&mut self, code: u32, bits: u32) {
        // Huffman codes are MSB-first per RFC 1951; reverse to LSB-first.
        let mut reversed = 0u32;
        for i in 0..bits {
            if (code >> i) & 1 != 0 {
                reversed |= 1 << (bits - 1 - i);
            }
        }
        self.write_bits(reversed, bits);
    }
    fn bit_count(&self) -> u32 {
        self.total
    }
    fn into_bytes(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.out.push((self.accum & 0xff) as u8);
        }
        self.out
    }
}

// ============================================================
// Bind-group helpers
// ============================================================

fn bge<'a>(binding: u32, buf: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
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
    use super::super::lz77::Token;
    use super::*;
    use flate2::read::DeflateDecoder;
    use std::io::Read;

    fn try_pipeline() -> Option<HuffmanEmitV2Pipeline> {
        let ctx = GpuContext::try_init().ok()?;
        Some(HuffmanEmitV2Pipeline::new(Arc::new(ctx)))
    }

    fn reconstruct(tokens: &[Token]) -> Vec<u8> {
        let mut out = Vec::new();
        for t in tokens {
            if t.is_literal() {
                out.push(t.distance as u8);
            } else {
                let start = out.len() - t.distance as usize;
                for i in 0..t.length as usize {
                    let b = out[start + i];
                    out.push(b);
                }
            }
        }
        out
    }

    fn round_trip(tokens: &[Token]) {
        let Some(p) = try_pipeline() else {
            eprintln!("no GPU — skipping");
            return;
        };
        let deflate_bytes = p.emit_fixed_block_v2(tokens);
        let mut decoded = Vec::new();
        DeflateDecoder::new(&deflate_bytes[..])
            .read_to_end(&mut decoded)
            .expect("flate2 should decode v2 GPU-emitted DEFLATE");
        let expected = reconstruct(tokens);
        assert_eq!(
            decoded,
            expected,
            "v2 GPU round-trip mismatch on {} tokens",
            tokens.len()
        );
    }

    #[test]
    fn v2_literals_only() {
        let tokens: Vec<Token> = (0..64u8).map(Token::literal).collect();
        round_trip(&tokens);
    }

    #[test]
    fn v2_single_back_ref() {
        let mut tokens: Vec<Token> = b"hello, ".iter().map(|&b| Token::literal(b)).collect();
        tokens.push(Token::back_ref(5, 7));
        round_trip(&tokens);
    }

    #[test]
    fn v2_many_back_refs_with_extra_bits() {
        let mut tokens: Vec<Token> = Vec::new();
        for i in 0..512u32 {
            tokens.push(Token::literal((i & 0xff) as u8));
        }
        for &(len, dist) in &[(3, 1), (11, 5), (35, 17), (131, 257)] {
            tokens.push(Token::back_ref(len, dist));
        }
        round_trip(&tokens);
    }

    #[test]
    fn v2_long_random_literal_stream() {
        let tokens: Vec<Token> = (0..2048u32)
            .map(|i| Token::literal((i.wrapping_mul(2654435761) as u32 & 0xff) as u8))
            .collect();
        round_trip(&tokens);
    }

    #[test]
    fn v2_empty_token_stream() {
        let Some(p) = try_pipeline() else { return };
        let deflate_bytes = p.emit_fixed_block_v2(&[]);
        let mut decoded = Vec::new();
        DeflateDecoder::new(&deflate_bytes[..])
            .read_to_end(&mut decoded)
            .expect("v2 empty fixed block should decode");
        assert!(decoded.is_empty());
    }

    /// Cross multi-workgroup boundary (>256 tokens) so the inter-workgroup
    /// scan is exercised, not just the local one.
    #[test]
    fn v2_multi_workgroup() {
        let tokens: Vec<Token> = (0..4096u32)
            .map(|i| Token::literal((i & 0xff) as u8))
            .collect();
        round_trip(&tokens);
    }

    /// D-3: dynamic Huffman round-trips. Same shader as v2 (fixed),
    /// just with dynamically-built per-block code lengths and a longer
    /// header. Output must be valid DEFLATE → reconstructs to original.
    fn round_trip_v3(tokens: &[Token]) {
        let Some(p) = try_pipeline() else {
            eprintln!("no GPU — skipping v3");
            return;
        };
        let deflate_bytes = p
            .emit_dynamic_block_v3(tokens)
            .expect("dynamic Huffman build");
        let mut decoded = Vec::new();
        DeflateDecoder::new(&deflate_bytes[..])
            .read_to_end(&mut decoded)
            .expect("flate2 should decode v3 GPU-emitted DEFLATE");
        let expected = reconstruct(tokens);
        assert_eq!(
            decoded,
            expected,
            "v3 GPU dynamic round-trip mismatch on {} tokens",
            tokens.len()
        );
    }

    #[test]
    fn v3_literals_only() {
        let tokens: Vec<Token> = (0..64u8).map(Token::literal).collect();
        round_trip_v3(&tokens);
    }

    #[test]
    fn v3_single_back_ref() {
        let mut tokens: Vec<Token> = b"hello, ".iter().map(|&b| Token::literal(b)).collect();
        tokens.push(Token::back_ref(5, 7));
        round_trip_v3(&tokens);
    }

    #[test]
    fn v3_many_back_refs_with_extra_bits() {
        let mut tokens: Vec<Token> = Vec::new();
        for i in 0..512u32 {
            tokens.push(Token::literal((i & 0xff) as u8));
        }
        for &(len, dist) in &[(3, 1), (11, 5), (35, 17), (131, 257)] {
            tokens.push(Token::back_ref(len, dist));
        }
        round_trip_v3(&tokens);
    }

    #[test]
    fn v3_skewed_frequencies() {
        // Heavy bias toward a few literal values exercises the
        // length-limited Huffman path more than uniform input does.
        let mut tokens: Vec<Token> = Vec::new();
        for _ in 0..2000 {
            tokens.push(Token::literal(b'x'));
        }
        for i in 0..32u32 {
            tokens.push(Token::literal((i & 0xff) as u8));
        }
        round_trip_v3(&tokens);
    }

    #[test]
    fn v3_multi_workgroup() {
        // Realistic-ish: many literals across workgroup boundary.
        let tokens: Vec<Token> = (0..4096u32)
            .map(|i| Token::literal((i & 0xff) as u8))
            .collect();
        round_trip_v3(&tokens);
    }
}
