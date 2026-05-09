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

use super::lz77::Token;

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

/// Encode `tokens` as a single DEFLATE block (BFINAL=1, BTYPE=01). Returns
/// the raw deflate bitstream.
pub fn encode_block(tokens: &[Token]) -> io::Result<Vec<u8>> {
    let mut bw = BitWriter::new(Vec::new());
    // Block header: BFINAL=1, BTYPE=01 (fixed). LSB-first within the byte
    // means the encoded order is 1, 1, 0 → bits 0b011 written low to high.
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

    // End-of-block (symbol 256).
    let (eob, eob_bits) = fixed_litlen_code(256);
    bw.write_huffman(eob, eob_bits)?;

    bw.finish()
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
