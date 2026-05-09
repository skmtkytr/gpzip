//! Encode an LZ77 token stream into a DEFLATE bitstream and wrap it as a
//! gzip member (RFC 1951 + RFC 1952).
//!
//! Uses BTYPE=01 (fixed Huffman). Slightly worse compression ratio than
//! BTYPE=10 (dynamic Huffman) but no table to compute and ship — keeps the
//! CPU-side encoder small.
//!
//! Verified end-to-end: `flate2`'s `MultiGzDecoder` reads back the bytes
//! produced here and returns the original input.

use std::io::{self, Write};

use super::huffman::{canonical_codes, try_build_code_lengths};
use super::lz77::Token;

/// RFC 1951 §3.2.7: order in which the meta-Huffman code lengths are written.
const META_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// Bit writer that packs bits LSB-first within each byte (DEFLATE convention).
struct BitWriter<W: Write> {
    inner: W,
    accum: u32,
    nbits: u32,
}

impl<W: Write> BitWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            accum: 0,
            nbits: 0,
        }
    }

    fn write_bits(&mut self, value: u32, count: u32) -> io::Result<()> {
        debug_assert!(count <= 24);
        self.accum |= (value & ((1u32 << count) - 1)) << self.nbits;
        self.nbits += count;
        while self.nbits >= 8 {
            let byte = (self.accum & 0xff) as u8;
            self.inner.write_all(&[byte])?;
            self.accum >>= 8;
            self.nbits -= 8;
        }
        Ok(())
    }

    /// Huffman codes are specified MSB-first. Reverse them to LSB-first
    /// before packing.
    fn write_huffman(&mut self, code: u32, bits: u32) -> io::Result<()> {
        let mut reversed = 0u32;
        for i in 0..bits {
            if (code >> i) & 1 != 0 {
                reversed |= 1 << (bits - 1 - i);
            }
        }
        self.write_bits(reversed, bits)
    }

    fn flush_byte(&mut self) -> io::Result<()> {
        if self.nbits > 0 {
            let byte = (self.accum & 0xff) as u8;
            self.inner.write_all(&[byte])?;
            self.accum = 0;
            self.nbits = 0;
        }
        Ok(())
    }

    fn finish(mut self) -> io::Result<W> {
        self.flush_byte()?;
        Ok(self.inner)
    }
}

/// Fixed Huffman code for a literal/length symbol (RFC 1951 §3.2.6).
fn fixed_litlen_code(symbol: u32) -> (u32, u32) {
    match symbol {
        0..=143 => (0b0011_0000 + symbol, 8),
        144..=255 => (0b1_1001_0000 + (symbol - 144), 9),
        256..=279 => (symbol - 256, 7),
        280..=287 => (0b1100_0000 + (symbol - 280), 8),
        _ => unreachable!(),
    }
}

/// Distance symbols use a fixed 5-bit code (RFC 1951 §3.2.6).
fn fixed_dist_code(symbol: u32) -> (u32, u32) {
    (symbol, 5)
}

/// Map a length 3..=258 to (length-symbol, extra-bits, extra-value).
/// Table per RFC 1951 §3.2.5.
fn length_code(length: u32) -> (u32, u32, u32) {
    debug_assert!((3..=258).contains(&length));
    // (base, extra, code)
    const TABLE: &[(u32, u32, u32)] = &[
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
    // Find the row whose `base` is the largest <= length.
    let mut idx = 0;
    for (i, row) in TABLE.iter().enumerate() {
        if row.0 <= length {
            idx = i;
        } else {
            break;
        }
    }
    let (base, extra, code) = TABLE[idx];
    (code, extra, length - base)
}

/// Map a distance 1..=32768 to (distance-symbol, extra-bits, extra-value).
/// Table per RFC 1951 §3.2.5.
fn distance_code(distance: u32) -> (u32, u32, u32) {
    debug_assert!((1..=32768).contains(&distance));
    const TABLE: &[(u32, u32, u32)] = &[
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
    let mut idx = 0;
    for (i, row) in TABLE.iter().enumerate() {
        if row.0 <= distance {
            idx = i;
        } else {
            break;
        }
    }
    let (base, extra, code) = TABLE[idx];
    (code, extra, distance - base)
}

/// Encode `tokens` as a single DEFLATE block. Tries dynamic Huffman
/// (BTYPE=10) first; falls back to fixed Huffman (BTYPE=01) if the natural
/// tree exceeds DEFLATE's 15-bit code length cap.
///
/// Retained as the A/B baseline against `encode_block_fast`; production
/// (`gpu.rs` chunk path) calls the fast version.
#[allow(dead_code)]
pub fn encode_block(tokens: &[Token]) -> io::Result<Vec<u8>> {
    let mut bw = BitWriter::new(Vec::new());
    if !try_write_dynamic_block(tokens, &mut bw)? {
        // Either tree was too tall or another constraint failed — fall back.
        bw = BitWriter::new(Vec::new());
        write_fixed_block(tokens, &mut bw)?;
    }
    bw.finish()
}

/// Encode `tokens` as a single DEFLATE block using fixed Huffman (BTYPE=01).
/// Kept for tests and as a fallback for tiny inputs where the dynamic header
/// outweighs its savings.
#[allow(dead_code)]
pub fn encode_fixed_block(tokens: &[Token]) -> io::Result<Vec<u8>> {
    let mut bw = BitWriter::new(Vec::new());
    write_fixed_block(tokens, &mut bw)?;
    bw.finish()
}

fn write_fixed_block<W: Write>(tokens: &[Token], bw: &mut BitWriter<W>) -> io::Result<()> {
    // BFINAL=1, BTYPE=01 → 0b011 LSB-first.
    bw.write_bits(0b011, 3)?;
    for tok in tokens {
        if tok.is_literal() {
            let (code, bits) = fixed_litlen_code(tok.distance);
            bw.write_huffman(code, bits)?;
        } else {
            let (lcode, lextra_bits, lextra_val) = length_code(tok.length);
            let (lhuf, lhuf_bits) = fixed_litlen_code(lcode);
            bw.write_huffman(lhuf, lhuf_bits)?;
            if lextra_bits > 0 {
                bw.write_bits(lextra_val, lextra_bits)?;
            }
            let (dcode, dextra_bits, dextra_val) = distance_code(tok.distance);
            let (dhuf, dhuf_bits) = fixed_dist_code(dcode);
            bw.write_huffman(dhuf, dhuf_bits)?;
            if dextra_bits > 0 {
                bw.write_bits(dextra_val, dextra_bits)?;
            }
        }
    }
    let (eob, eob_bits) = fixed_litlen_code(256);
    bw.write_huffman(eob, eob_bits)?;
    Ok(())
}

/// Single RLE entry for the code-length sequence (RFC 1951 §3.2.7).
struct RleEntry {
    code: u8,
    extra_bits: u32,
    extra_value: u32,
}

/// RLE-encode a sequence of code lengths using the meta-alphabet symbols
/// 0..18 (RFC 1951 §3.2.7).
fn rle_encode_lengths(lens: &[u8]) -> Vec<RleEntry> {
    let mut out = Vec::with_capacity(lens.len());
    let mut i = 0;
    while i < lens.len() {
        let cur = lens[i];
        // Run length: how many consecutive entries equal cur.
        let mut run = 1usize;
        while i + run < lens.len() && lens[i + run] == cur {
            run += 1;
        }
        if cur == 0 {
            // Use 18 (11..138 zeros), then 17 (3..10 zeros), then literal 0.
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
            // Emit the value once literally, then 16 (3..6 repeats of prev).
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

/// Trim trailing zeros from a code-length array, but keep at least `min` entries.
fn trim_to_min(lens: &[u8], min: usize) -> usize {
    let mut n = lens.len();
    while n > min && lens[n - 1] == 0 {
        n -= 1;
    }
    n
}

/// Try to write a dynamic-Huffman block. Returns `Ok(true)` on success;
/// `Ok(false)` if the natural Huffman tree exceeds DEFLATE's 15-bit limit
/// and the caller should fall back to the fixed-Huffman block.
#[allow(dead_code)]
fn try_write_dynamic_block<W: Write>(tokens: &[Token], bw: &mut BitWriter<W>) -> io::Result<bool> {
    // 1. Frequencies. Always count EOB at least once.
    let mut litlen_freq = [0u32; 286];
    let mut dist_freq = [0u32; 30];
    for tok in tokens {
        if tok.is_literal() {
            litlen_freq[tok.distance as usize] += 1;
        } else {
            let (lcode, _, _) = length_code(tok.length);
            litlen_freq[lcode as usize] += 1;
            let (dcode, _, _) = distance_code(tok.distance);
            dist_freq[dcode as usize] += 1;
        }
    }
    litlen_freq[256] += 1; // EOB

    // 2. Length-limited Huffman (15 bits both alphabets). Fall back to fixed
    // if the natural tree is taller than DEFLATE allows.
    let Some(mut litlen_lens) = try_build_code_lengths(&litlen_freq, 15) else {
        return Ok(false);
    };
    let Some(mut dist_lens) = try_build_code_lengths(&dist_freq, 15) else {
        return Ok(false);
    };

    // DEFLATE quirk: if there's only one (or zero) distance code, you must
    // still emit at least one with length=1 and pad the table to two
    // entries — the decoder needs a non-empty distance Huffman tree.
    let used_dist: usize = dist_lens.iter().filter(|&&l| l > 0).count();
    if used_dist == 0 {
        dist_lens[0] = 1;
        dist_lens[1] = 1;
    } else if used_dist == 1 {
        // Find the used one; if it's index 0, also set index 1; else set index 0.
        let only = dist_lens.iter().position(|&l| l > 0).unwrap();
        dist_lens[only] = 1;
        dist_lens[1 - only.min(1)] = 1;
    }

    // Same edge for litlen (extremely unlikely — EOB is always present).
    if litlen_lens.iter().filter(|&&l| l > 0).count() < 2 {
        // Force code 0 and EOB to length 1 each.
        litlen_lens[0] = 1;
        litlen_lens[256] = 1;
    }

    let litlen_codes = canonical_codes(&litlen_lens);
    let dist_codes = canonical_codes(&dist_lens);

    // 3. Determine HLIT, HDIST.
    let nlit = trim_to_min(&litlen_lens, 257);
    let ndist = trim_to_min(&dist_lens, 1);
    let hlit = (nlit - 257) as u32;
    let hdist = (ndist - 1) as u32;

    // 4. RLE-encode the combined length sequence.
    let mut combined = Vec::with_capacity(nlit + ndist);
    combined.extend_from_slice(&litlen_lens[..nlit]);
    combined.extend_from_slice(&dist_lens[..ndist]);
    let rle = rle_encode_lengths(&combined);

    // 5. Meta-Huffman from RLE symbol frequencies.
    let mut meta_freq = [0u32; 19];
    for e in &rle {
        meta_freq[e.code as usize] += 1;
    }
    let Some(meta_lens) = try_build_code_lengths(&meta_freq, 7) else {
        return Ok(false);
    };
    let meta_codes = canonical_codes(&meta_lens);

    // 6. Determine HCLEN — number of meta-Huffman code lengths to write,
    // in the special order. Trim trailing zeros but keep at least 4.
    let mut hclen_count = 19;
    while hclen_count > 4 && meta_lens[META_ORDER[hclen_count - 1]] == 0 {
        hclen_count -= 1;
    }
    let hclen = (hclen_count - 4) as u32;

    // 7. Write block header.
    // BFINAL=1, BTYPE=10 → bits 1, 0, 1 written LSB-first → value 0b101.
    bw.write_bits(0b101, 3)?;
    bw.write_bits(hlit, 5)?;
    bw.write_bits(hdist, 5)?;
    bw.write_bits(hclen, 4)?;

    // 8. Meta-Huffman code lengths (3 bits each, in special order).
    for i in 0..hclen_count {
        bw.write_bits(meta_lens[META_ORDER[i]] as u32, 3)?;
    }

    // 9. Encoded code lengths (RLE through meta-Huffman).
    for e in &rle {
        let code = meta_codes[e.code as usize];
        let bits = meta_lens[e.code as usize] as u32;
        bw.write_huffman(code, bits)?;
        if e.extra_bits > 0 {
            bw.write_bits(e.extra_value, e.extra_bits)?;
        }
    }

    // 10. Emit tokens with dynamic codes.
    for tok in tokens {
        if tok.is_literal() {
            let sym = tok.distance as usize;
            bw.write_huffman(litlen_codes[sym], litlen_lens[sym] as u32)?;
        } else {
            let (lcode, lex_bits, lex_val) = length_code(tok.length);
            bw.write_huffman(
                litlen_codes[lcode as usize],
                litlen_lens[lcode as usize] as u32,
            )?;
            if lex_bits > 0 {
                bw.write_bits(lex_val, lex_bits)?;
            }
            let (dcode, dex_bits, dex_val) = distance_code(tok.distance);
            bw.write_huffman(dist_codes[dcode as usize], dist_lens[dcode as usize] as u32)?;
            if dex_bits > 0 {
                bw.write_bits(dex_val, dex_bits)?;
            }
        }
    }

    // 11. End-of-block.
    bw.write_huffman(litlen_codes[256], litlen_lens[256] as u32)?;
    Ok(true)
}

// ============================================================
// Optimized encoder (encode_block_fast).
//
// Same DEFLATE bitstream contract as `encode_block`, but redesigned for
// throughput. Three changes carry the speedup:
//
// 1. Huffman codes are bit-reversed *once* per block (286 + 30 + 19 = 335
//    entries) and cached, instead of reversing per emit. The original
//    `BitWriter::write_huffman` ran a per-bit loop on every literal.
//
// 2. The bit accumulator is u64 and the output is a direct `Vec<u8>` push,
//    not a `Write::write_all(&[byte])` per emitted byte. Eliminates the
//    Write trait dispatch and the per-byte allocation pattern.
//
// 3. Length and distance symbol lookup is a direct table index instead of
//    a linear scan over the 29/30-row table. `LEN_LUT[length]` and
//    `DIST_LUT_LO/HI` (zlib-style 256+512 split) hit O(1).
//
// Per-stage profiling on a 512 KiB random chunk attributed 11.7 ms of
// 12.8 ms total to `encode_block`; this rewrite was the largest available
// per-chunk win in the GPU pipeline.
// ============================================================

use std::sync::OnceLock;

/// Precomputed (sym, extra_bits, base) for length 3..=258. Index = length.
/// Slots 0..3 are unused (length must be >= 3).
struct LenEntry {
    sym: u16,
    extra_bits: u8,
    base: u16,
}

fn len_lut() -> &'static [LenEntry; 259] {
    static LUT: OnceLock<[LenEntry; 259]> = OnceLock::new();
    LUT.get_or_init(|| {
        // Same table as length_code(), inlined here to drive the LUT build.
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
        // Build by walking ROWS once and filling each row's range.
        let mut out: [LenEntry; 259] = std::array::from_fn(|_| LenEntry {
            sym: 0,
            extra_bits: 0,
            base: 0,
        });
        for w in 0..ROWS.len() {
            let (base, extra, sym) = ROWS[w];
            // Range covered by this row: [base, next_base)
            let end = if w + 1 < ROWS.len() {
                ROWS[w + 1].0
            } else {
                259
            };
            let mut len = base;
            while len < end {
                out[len as usize] = LenEntry {
                    sym: sym as u16,
                    extra_bits: extra as u8,
                    base: base as u16,
                };
                len += 1;
            }
        }
        out
    })
}

/// Same shape for distance, but distance space is 1..=32768 — too big for a
/// flat table. Use the zlib trick: distances 1..=256 hit DIST_LO directly;
/// larger distances hit DIST_HI indexed by `(dist-1) >> 7` (which yields
/// 256..=511, 256 entries covering the upper 30K-distance range).
struct DistEntry {
    sym: u16,
    extra_bits: u8,
    base: u16,
}

fn dist_lut_lo() -> &'static [DistEntry; 257] {
    static LUT: OnceLock<[DistEntry; 257]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut out: [DistEntry; 257] = std::array::from_fn(|_| DistEntry {
            sym: 0,
            extra_bits: 0,
            base: 0,
        });
        let rows: &[(u32, u32, u32)] = &[
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
        ];
        // Last row (257..) belongs in DIST_HI; here we fill 1..=256.
        for w in 0..rows.len() {
            let (base, extra, sym) = rows[w];
            let end = if w + 1 < rows.len() {
                rows[w + 1].0
            } else {
                257
            };
            let mut d = base;
            while d < end {
                out[d as usize] = DistEntry {
                    sym: sym as u16,
                    extra_bits: extra as u8,
                    base: base as u16,
                };
                d += 1;
            }
        }
        out
    })
}

fn dist_lut_hi() -> &'static [DistEntry; 256] {
    static LUT: OnceLock<[DistEntry; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let rows: &[(u32, u32, u32)] = &[
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
        // Index space: ((dist - 1) >> 7) for dist >= 257 → starts at index 2.
        // We allocate a 256-entry table so that even maximum distance 32768
        // → index ((32768-1) >> 7) = 255 fits.
        let mut out: [DistEntry; 256] = std::array::from_fn(|_| DistEntry {
            sym: 0,
            extra_bits: 0,
            base: 0,
        });
        for w in 0..rows.len() {
            let (base, extra, sym) = rows[w];
            let end = if w + 1 < rows.len() {
                rows[w + 1].0
            } else {
                32769
            };
            let mut d = base;
            while d < end {
                let idx = ((d - 1) >> 7) as usize;
                if idx < out.len() {
                    out[idx] = DistEntry {
                        sym: sym as u16,
                        extra_bits: extra as u8,
                        base: base as u16,
                    };
                }
                d += 1;
            }
        }
        out
    })
}

#[inline(always)]
fn dist_entry(distance: u32) -> &'static DistEntry {
    if distance <= 256 {
        &dist_lut_lo()[distance as usize]
    } else {
        &dist_lut_hi()[((distance - 1) >> 7) as usize]
    }
}

/// Reverse the low `bits` bits of `code`. Uses the hardware bit-reverse
/// instruction (single insn on x86 BMI / ARM RBIT) instead of a loop.
#[inline(always)]
fn reverse_bits(code: u32, bits: u32) -> u32 {
    if bits == 0 {
        0
    } else {
        code.reverse_bits() >> (32 - bits)
    }
}

/// u64 bit accumulator + direct Vec<u8> push. The original `BitWriter`
/// accumulator was u32 and emitted via `Write::write_all(&[byte])` per byte;
/// both kept it correct but neither was free in a tight emit loop.
struct VecBitWriter {
    out: Vec<u8>,
    accum: u64,
    nbits: u32,
}

impl VecBitWriter {
    fn new(capacity: usize) -> Self {
        Self {
            out: Vec::with_capacity(capacity),
            accum: 0,
            nbits: 0,
        }
    }

    /// Write up to 32 bits. The caller must ensure `count <= 32`.
    #[inline(always)]
    fn write_bits(&mut self, value: u32, count: u32) {
        debug_assert!(count <= 32);
        self.accum |= (value as u64) << self.nbits;
        self.nbits += count;
        while self.nbits >= 8 {
            self.out.push(self.accum as u8);
            self.accum >>= 8;
            self.nbits -= 8;
        }
    }

    /// Write an already-reversed Huffman code. Identical to `write_bits`
    /// but a separate name documents that the caller did the reversal at
    /// table-build time.
    #[inline(always)]
    fn write_pre(&mut self, code_pre: u32, bits: u32) {
        self.accum |= (code_pre as u64) << self.nbits;
        self.nbits += bits;
        while self.nbits >= 8 {
            self.out.push(self.accum as u8);
            self.accum >>= 8;
            self.nbits -= 8;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.out.push(self.accum as u8);
        }
        self.out
    }
}

/// Optimized counterpart of `encode_block`. Produces a DEFLATE bitstream
/// that decodes to the same bytes (verified by the round-trip tests
/// below). Bit-identical output is *not* guaranteed because the canonical
/// Huffman algorithm and the RLE choices are unchanged but the encoding
/// order through the new accumulator may shift trailing-byte padding bits.
pub fn encode_block_fast(tokens: &[Token]) -> io::Result<Vec<u8>> {
    // Estimate ~5/8 bits per literal as a reasonable initial Vec capacity.
    let est = tokens.len().max(64);
    let mut bw = VecBitWriter::new(est);
    if !try_write_dynamic_block_fast(tokens, &mut bw) {
        bw = VecBitWriter::new(est);
        write_fixed_block_fast(tokens, &mut bw);
    }
    Ok(bw.finish())
}

fn write_fixed_block_fast(tokens: &[Token], bw: &mut VecBitWriter) {
    // BFINAL=1, BTYPE=01.
    bw.write_bits(0b011, 3);

    // Precompute reversed fixed Huffman codes once. Fixed codes are
    // deterministic — could be a `const` table, but lazy via OnceLock keeps
    // this section self-contained.
    let fixed = fixed_litlen_pre();
    let fixed_dist = fixed_dist_pre();
    let len_t = len_lut();

    for tok in tokens {
        if tok.is_literal() {
            let sym = tok.distance as usize;
            let (pre, bits) = fixed[sym];
            bw.write_pre(pre, bits);
        } else {
            let le = &len_t[tok.length as usize];
            let (pre, bits) = fixed[le.sym as usize];
            bw.write_pre(pre, bits);
            let extra = tok.length as u16 - le.base;
            if le.extra_bits > 0 {
                bw.write_bits(extra as u32, le.extra_bits as u32);
            }
            let de = dist_entry(tok.distance);
            // Distance fixed code = symbol in 5 bits, MSB-first → reverse.
            let (dpre, dbits) = fixed_dist[de.sym as usize];
            bw.write_pre(dpre, dbits);
            let dextra = tok.distance - de.base as u32;
            if de.extra_bits > 0 {
                bw.write_bits(dextra, de.extra_bits as u32);
            }
        }
    }
    let (eob, eob_bits) = fixed[256];
    bw.write_pre(eob, eob_bits);
}

fn fixed_litlen_pre() -> &'static [(u32, u32); 288] {
    static LUT: OnceLock<[(u32, u32); 288]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut out = [(0u32, 0u32); 288];
        for sym in 0..288u32 {
            let (code, bits) = match sym {
                0..=143 => (0b0011_0000 + sym, 8),
                144..=255 => (0b1_1001_0000 + (sym - 144), 9),
                256..=279 => (sym - 256, 7),
                280..=287 => (0b1100_0000 + (sym - 280), 8),
                _ => unreachable!(),
            };
            out[sym as usize] = (reverse_bits(code, bits), bits);
        }
        out
    })
}

fn fixed_dist_pre() -> &'static [(u32, u32); 30] {
    static LUT: OnceLock<[(u32, u32); 30]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut out = [(0u32, 0u32); 30];
        for sym in 0..30u32 {
            out[sym as usize] = (reverse_bits(sym, 5), 5);
        }
        out
    })
}

fn try_write_dynamic_block_fast(tokens: &[Token], bw: &mut VecBitWriter) -> bool {
    // Frequency pass — same logic as the original.
    let len_t = len_lut();
    let mut litlen_freq = [0u32; 286];
    let mut dist_freq = [0u32; 30];
    for tok in tokens {
        if tok.is_literal() {
            litlen_freq[tok.distance as usize] += 1;
        } else {
            let le = &len_t[tok.length as usize];
            litlen_freq[le.sym as usize] += 1;
            let de = dist_entry(tok.distance);
            dist_freq[de.sym as usize] += 1;
        }
    }
    litlen_freq[256] += 1;

    let Some(mut litlen_lens) = try_build_code_lengths(&litlen_freq, 15) else {
        return false;
    };
    let Some(mut dist_lens) = try_build_code_lengths(&dist_freq, 15) else {
        return false;
    };

    // Same single-distance edge-case fixup as the original.
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

    // The big win: pre-reverse every code we'll emit, once.
    let mut litlen_pre = vec![0u32; litlen_codes.len()];
    for i in 0..litlen_codes.len() {
        if litlen_lens[i] > 0 {
            litlen_pre[i] = reverse_bits(litlen_codes[i], litlen_lens[i] as u32);
        }
    }
    let mut dist_pre = vec![0u32; dist_codes.len()];
    for i in 0..dist_codes.len() {
        if dist_lens[i] > 0 {
            dist_pre[i] = reverse_bits(dist_codes[i], dist_lens[i] as u32);
        }
    }

    let nlit = trim_to_min(&litlen_lens, 257);
    let ndist = trim_to_min(&dist_lens, 1);
    let hlit = (nlit - 257) as u32;
    let hdist = (ndist - 1) as u32;

    let mut combined = Vec::with_capacity(nlit + ndist);
    combined.extend_from_slice(&litlen_lens[..nlit]);
    combined.extend_from_slice(&dist_lens[..ndist]);
    let rle = rle_encode_lengths(&combined);

    let mut meta_freq = [0u32; 19];
    for e in &rle {
        meta_freq[e.code as usize] += 1;
    }
    let Some(meta_lens) = try_build_code_lengths(&meta_freq, 7) else {
        return false;
    };
    let meta_codes = canonical_codes(&meta_lens);
    let mut meta_pre = [0u32; 19];
    for i in 0..19 {
        if meta_lens[i] > 0 {
            meta_pre[i] = reverse_bits(meta_codes[i], meta_lens[i] as u32);
        }
    }

    let mut hclen_count = 19;
    while hclen_count > 4 && meta_lens[META_ORDER[hclen_count - 1]] == 0 {
        hclen_count -= 1;
    }
    let hclen = (hclen_count - 4) as u32;

    bw.write_bits(0b101, 3);
    bw.write_bits(hlit, 5);
    bw.write_bits(hdist, 5);
    bw.write_bits(hclen, 4);

    for i in 0..hclen_count {
        bw.write_bits(meta_lens[META_ORDER[i]] as u32, 3);
    }

    for e in &rle {
        let bits = meta_lens[e.code as usize] as u32;
        bw.write_pre(meta_pre[e.code as usize], bits);
        if e.extra_bits > 0 {
            bw.write_bits(e.extra_value, e.extra_bits);
        }
    }

    // Hot loop. Direct table indices, no per-emit bit reversal.
    for tok in tokens {
        if tok.is_literal() {
            let sym = tok.distance as usize;
            bw.write_pre(litlen_pre[sym], litlen_lens[sym] as u32);
        } else {
            let le = &len_t[tok.length as usize];
            let sym = le.sym as usize;
            bw.write_pre(litlen_pre[sym], litlen_lens[sym] as u32);
            if le.extra_bits > 0 {
                let extra = tok.length as u16 - le.base;
                bw.write_bits(extra as u32, le.extra_bits as u32);
            }
            let de = dist_entry(tok.distance);
            let dsym = de.sym as usize;
            bw.write_pre(dist_pre[dsym], dist_lens[dsym] as u32);
            if de.extra_bits > 0 {
                let dextra = tok.distance - de.base as u32;
                bw.write_bits(dextra, de.extra_bits as u32);
            }
        }
    }

    bw.write_pre(litlen_pre[256], litlen_lens[256] as u32);
    true
}

/// Wrap a raw DEFLATE bitstream as a gzip member (RFC 1952).
pub fn gzip_wrap(deflate: &[u8], original: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(deflate.len() + 18);
    // Header
    out.extend_from_slice(&[
        0x1f, 0x8b, // magic
        0x08, // CM = deflate
        0x00, // FLG = none
        0x00, 0x00, 0x00, 0x00, // MTIME = 0
        0x00, // XFL
        0xff, // OS = unknown
    ]);
    out.extend_from_slice(deflate);
    let crc = crc32fast::hash(original);
    out.extend_from_slice(&crc.to_le_bytes());
    // ISIZE: original size mod 2^32. The cast already takes the low 32 bits.
    let isize = original.len() as u32;
    out.extend_from_slice(&isize.to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::read::MultiGzDecoder;
    use std::io::Read;

    use crate::gpu::lz77::{greedy_walk, Token};

    /// CPU LZ77: brute-force greedy match find. Produces the same per-position
    /// output as the GPU shader (used to test the encoder without a GPU).
    fn cpu_lz77(input: &[u8], window: usize) -> Vec<Token> {
        let mut out = Vec::with_capacity(input.len());
        for pos in 0..input.len() {
            let mut best_len = 0usize;
            let mut best_dist = 0usize;
            let start = pos.saturating_sub(window);
            for prev in start..pos {
                let mut len = 0;
                while pos + len < input.len() && len < 258 && input[prev + len] == input[pos + len]
                {
                    len += 1;
                }
                if len > best_len {
                    best_len = len;
                    best_dist = pos - prev;
                }
            }
            if best_len >= 3 {
                out.push(Token::back_ref(best_len as u32, best_dist as u32));
            } else {
                out.push(Token::literal(input[pos]));
            }
        }
        out
    }

    fn round_trip(input: &[u8]) {
        let raw = cpu_lz77(input, 4096);
        let walked = greedy_walk(&raw, input);
        let deflate = encode_block(&walked).expect("encode");
        let gzipped = gzip_wrap(&deflate, input);
        let mut decoded = Vec::new();
        MultiGzDecoder::new(&gzipped[..])
            .read_to_end(&mut decoded)
            .expect("decode");
        assert_eq!(decoded, input, "round trip mismatch");
    }

    #[test]
    fn empty_input() {
        round_trip(b"");
    }

    #[test]
    fn single_byte() {
        round_trip(b"a");
    }

    #[test]
    fn no_matches() {
        let input: Vec<u8> = (0..32u8).collect();
        round_trip(&input);
    }

    #[test]
    fn obvious_repeat() {
        round_trip(&b"abcabcabcabcabcabcabc".repeat(8));
    }

    #[test]
    fn self_overlapping_match() {
        round_trip(&[b'x'; 64]);
    }

    #[test]
    fn larger_realistic_data() {
        let mut input = Vec::new();
        for i in 0..256 {
            input.extend_from_slice(format!("line {i}: the quick brown fox\n").as_bytes());
        }
        round_trip(&input);
    }

    #[test]
    fn pseudo_random() {
        let input: Vec<u8> = (0..1500u32)
            .map(|i| (i.wrapping_mul(2654435761)) as u8)
            .collect();
        round_trip(&input);
    }

    /// Same round-trip as above but through the optimized encoder.
    fn round_trip_fast(input: &[u8]) {
        let raw = cpu_lz77(input, 4096);
        let walked = greedy_walk(&raw, input);
        let deflate = encode_block_fast(&walked).expect("encode_fast");
        let gzipped = gzip_wrap(&deflate, input);
        let mut decoded = Vec::new();
        MultiGzDecoder::new(&gzipped[..])
            .read_to_end(&mut decoded)
            .expect("decode");
        assert_eq!(decoded, input, "fast round trip mismatch");
    }

    #[test]
    fn fast_empty_input() {
        round_trip_fast(b"");
    }

    #[test]
    fn fast_single_byte() {
        round_trip_fast(b"a");
    }

    #[test]
    fn fast_no_matches() {
        let input: Vec<u8> = (0..32u8).collect();
        round_trip_fast(&input);
    }

    #[test]
    fn fast_obvious_repeat() {
        round_trip_fast(&b"abcabcabcabcabcabcabc".repeat(8));
    }

    #[test]
    fn fast_self_overlapping_match() {
        round_trip_fast(&[b'x'; 64]);
    }

    #[test]
    fn fast_larger_realistic_data() {
        let mut input = Vec::new();
        for i in 0..256 {
            input.extend_from_slice(format!("line {i}: the quick brown fox\n").as_bytes());
        }
        round_trip_fast(&input);
    }

    #[test]
    fn fast_pseudo_random() {
        let input: Vec<u8> = (0..1500u32)
            .map(|i| (i.wrapping_mul(2654435761)) as u8)
            .collect();
        round_trip_fast(&input);
    }

    /// A/B benchmark: feed the same real GPU-derived token stream into both
    /// encoders, measure separately, verify both round-trip to the original
    /// bytes. `#[ignore]` so normal `cargo test` skips it (it needs a GPU and
    /// runs much longer than a unit test). Invoke with:
    ///
    /// ```sh
    /// cargo test --release -p gpzip-codec-gpu --features enabled \
    ///     ab_encoder_benchmark -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn ab_encoder_benchmark() {
        use crate::gpu::context::GpuContext;
        use crate::gpu::lz77_hash::{Lz77HashPipeline, DEFAULT_WINDOW};
        use std::sync::Arc;
        use std::time::Instant;

        let Some(ctx) = GpuContext::try_init().ok() else {
            eprintln!("no GPU — skipping A/B");
            return;
        };
        let pipeline = Lz77HashPipeline::new(Arc::new(ctx));

        let chunk = 512 * 1024usize;
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
            "{:<6} {:>10} {:>10} {:>10}  {:>8} {:>8}  {:>8}",
            "wkld", "tokens", "v1_ms", "v2_ms", "v1_KiB", "v2_KiB", "speedup"
        );
        eprintln!("{}", "-".repeat(70));

        for (name, data) in &workloads {
            let raw = pipeline.match_find(data, DEFAULT_WINDOW);
            let walked = greedy_walk(&raw, data);
            let n_tokens = walked.len();

            // Warm up both encoders.
            for _ in 0..2 {
                let _ = encode_block(&walked).unwrap();
                let _ = encode_block_fast(&walked).unwrap();
            }

            let iters = 16;

            let t = Instant::now();
            let mut v1_out = Vec::new();
            for _ in 0..iters {
                v1_out = encode_block(&walked).unwrap();
            }
            let v1_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

            let t = Instant::now();
            let mut v2_out = Vec::new();
            for _ in 0..iters {
                v2_out = encode_block_fast(&walked).unwrap();
            }
            let v2_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

            // Correctness: both round-trip to original bytes after gzip-wrapping.
            let g1 = gzip_wrap(&v1_out, data);
            let g2 = gzip_wrap(&v2_out, data);
            let mut d1 = Vec::new();
            let mut d2 = Vec::new();
            MultiGzDecoder::new(&g1[..]).read_to_end(&mut d1).unwrap();
            MultiGzDecoder::new(&g2[..]).read_to_end(&mut d2).unwrap();
            assert_eq!(&d1, data, "v1 round-trip failed for {name}");
            assert_eq!(&d2, data, "v2 round-trip failed for {name}");

            let speedup = v1_ms / v2_ms;
            eprintln!(
                "{:<6} {:>10} {:>10.3} {:>10.3}  {:>8} {:>8}  {:>7.2}x",
                name,
                n_tokens,
                v1_ms,
                v2_ms,
                v1_out.len() / 1024,
                v2_out.len() / 1024,
                speedup
            );
        }
    }
}
