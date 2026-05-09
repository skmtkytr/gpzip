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
}
