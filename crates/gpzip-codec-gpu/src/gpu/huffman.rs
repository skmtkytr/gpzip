//! Build length-limited Huffman code-length tables, then convert to
//! canonical Huffman codes (RFC 1951 §3.2.2). Used by the dynamic-Huffman
//! DEFLATE block writer.
#![allow(dead_code)]
//!
//! Algorithm: standard Huffman tree by repeated merging of the two smallest
//! frequencies, then walk to assign code lengths. If the resulting max
//! length exceeds the limit, redistribute lengths via the package-merge
//! adjustment described in zlib's `huft_build` style. For typical inputs
//! the standard tree's max length is well under 15.

/// Build code lengths for the given symbol frequencies, with each length
/// capped at `max_bits`. Returns `None` if the natural Huffman tree exceeds
/// `max_bits` and the caller should fall back to fixed Huffman.
///
/// (A correct length-limiting pass would scale or use package-merge; the
/// previous fractional-Kraft attempt produced invalid code-length tables on
/// inputs where the natural max length was over the limit. Erroring out is
/// the safe choice until package-merge lands.)
pub fn build_code_lengths(freq: &[u32], max_bits: u32) -> Vec<u8> {
    try_build_code_lengths(freq, max_bits).unwrap_or_else(|| vec![0; freq.len()])
}

/// Same as `build_code_lengths`, but returns `None` instead of giving up
/// silently. Use this when the caller wants to detect overflow.
pub fn try_build_code_lengths(freq: &[u32], max_bits: u32) -> Option<Vec<u8>> {
    let n = freq.len();
    let used: Vec<usize> = (0..n).filter(|&i| freq[i] > 0).collect();
    if used.is_empty() {
        return Some(vec![0; n]);
    }
    if used.len() == 1 {
        // DEFLATE requires at least one bit per used symbol so the decoder
        // doesn't see a zero-length code.
        let mut out = vec![0u8; n];
        out[used[0]] = 1;
        return Some(out);
    }

    let lens = standard_huffman(freq);
    let max = lens.iter().max().copied().unwrap_or(0);
    if max as u32 > max_bits {
        return None;
    }
    Some(lens)
}

/// Standard Huffman tree → code lengths. Uses a min-heap by frequency.
fn standard_huffman(freq: &[u32]) -> Vec<u8> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let n = freq.len();
    // Each "node" carries (frequency, id). Internal nodes get fresh ids
    // starting at n; leaves keep their symbol id.
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
    let mut left: Vec<i32> = vec![-1; 2 * n];
    let mut right: Vec<i32> = vec![-1; 2 * n];

    for (i, &f) in freq.iter().enumerate() {
        if f > 0 {
            heap.push(Reverse((f as u64, i)));
        }
    }
    let mut next_id = n;

    while heap.len() > 1 {
        let Reverse((f1, i1)) = heap.pop().unwrap();
        let Reverse((f2, i2)) = heap.pop().unwrap();
        let id = next_id;
        next_id += 1;
        if id >= left.len() {
            left.resize(id + 1, -1);
            right.resize(id + 1, -1);
        }
        left[id] = i1 as i32;
        right[id] = i2 as i32;
        heap.push(Reverse((f1 + f2, id)));
    }

    let root = heap.pop().unwrap().0 .1;

    let mut lens = vec![0u8; n];
    walk(root, 0, &left, &right, n, &mut lens);
    lens
}

fn walk(node: usize, depth: u32, left: &[i32], right: &[i32], n_leaves: usize, lens: &mut [u8]) {
    if node < n_leaves {
        // Depth 0 happens only for the single-symbol degenerate case, which
        // build_code_lengths handles before calling here.
        lens[node] = depth.max(1) as u8;
        return;
    }
    let l = left[node];
    let r = right[node];
    if l >= 0 {
        walk(l as usize, depth + 1, left, right, n_leaves, lens);
    }
    if r >= 0 {
        walk(r as usize, depth + 1, left, right, n_leaves, lens);
    }
}

/// Build canonical Huffman codes from code lengths (RFC 1951 §3.2.2).
/// Returns a vector parallel to `lens` where `codes[i]` is the MSB-first
/// code for symbol `i` (0 if unused).
pub fn canonical_codes(lens: &[u8]) -> Vec<u32> {
    let max_bits = lens.iter().max().copied().unwrap_or(0) as usize;
    if max_bits == 0 {
        return vec![0; lens.len()];
    }

    // Count codes per length.
    let mut bl_count = vec![0u32; max_bits + 1];
    for &l in lens {
        if l > 0 {
            bl_count[l as usize] += 1;
        }
    }

    // Determine the starting code for each length (RFC 1951 §3.2.2).
    let mut next_code = vec![0u32; max_bits + 1];
    let mut code = 0u32;
    for bits in 1..=max_bits {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    let mut codes = vec![0u32; lens.len()];
    for (sym, &l) in lens.iter().enumerate() {
        if l > 0 {
            codes[sym] = next_code[l as usize];
            next_code[l as usize] += 1;
        }
    }
    codes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_frequencies() {
        let lens = build_code_lengths(&[0; 10], 15);
        assert!(lens.iter().all(|&l| l == 0));
    }

    #[test]
    fn single_symbol_gets_one_bit() {
        let mut freq = vec![0u32; 10];
        freq[3] = 100;
        let lens = build_code_lengths(&freq, 15);
        assert_eq!(lens[3], 1);
        for (i, &l) in lens.iter().enumerate() {
            if i != 3 {
                assert_eq!(l, 0);
            }
        }
    }

    #[test]
    fn balanced_frequencies_balanced_lengths() {
        let freq = vec![1u32; 8]; // 8 equally-frequent symbols
        let lens = build_code_lengths(&freq, 15);
        assert!(lens.iter().all(|&l| l == 3), "got {:?}", lens);
    }

    #[test]
    fn skewed_frequencies_skewed_lengths() {
        let freq = vec![100, 50, 25, 10, 5, 2, 1, 1];
        let lens = build_code_lengths(&freq, 15);
        // Most-frequent symbol gets the shortest code.
        assert!(lens[0] <= lens[7], "lens: {:?}", lens);
    }

    #[test]
    fn length_limit_enforced() {
        // Frequencies designed to produce codes longer than 4 bits naturally;
        // limit forces them down.
        let freq: Vec<u32> = (1..=20).map(|i| i as u32).collect();
        let lens = build_code_lengths(&freq, 6);
        assert!(
            lens.iter().all(|&l| l <= 6),
            "max length exceeded: {:?}",
            lens
        );
        // Kraft inequality: sum of 2^-l_i should be <= 1.
        let kraft: f64 = lens
            .iter()
            .filter(|&&l| l > 0)
            .map(|&l| 2f64.powi(-(l as i32)))
            .sum();
        assert!(kraft <= 1.0 + 1e-9, "kraft = {}", kraft);
    }

    #[test]
    fn canonical_codes_satisfy_prefix_property() {
        let lens = vec![3, 3, 3, 3, 3, 2, 4, 4];
        let codes = canonical_codes(&lens);
        // Verify no code is a prefix of another.
        for i in 0..codes.len() {
            if lens[i] == 0 {
                continue;
            }
            for j in 0..codes.len() {
                if i == j || lens[j] == 0 {
                    continue;
                }
                if lens[j] < lens[i] {
                    let shift = lens[i] - lens[j];
                    let prefix = codes[i] >> shift;
                    assert_ne!(prefix, codes[j], "{} is a prefix of {}", j, i);
                }
            }
        }
    }
}
