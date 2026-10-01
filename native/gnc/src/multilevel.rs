//! The multilevel traversal and predictor of `geoneural.codecs.multilevel`, decoder side.
//!
//! The stride-S0 sub-lattice is stored directly (zstd of zigzag first differences). Then for each stride s from
//! S0 down to 2, pass 0 predicts the new rows at s/2 along columns and pass 1 every remaining node of the s/2
//! lattice along rows, each from a 4 x 3 stencil of decoded values. Every float operation below is the float32
//! operation of `predict_pass`, in the same order: no fused multiply-add, square root in float32, rounding half
//! to even. A pass never reads its own targets, so each node is predicted and decoded in one step.

use crate::container::{Coder, Kind, Product};
use crate::model::Model;
use crate::rans::Decoder;
use crate::tables::{
    BINS, BINS_PER_OCTAVE, FEATURES, LATTICE_M_F32, LEVELS, LOG2_SCALE_MIN_F32, S0,
};
use crate::{fail, Result};
use std::io::Read;

/// Largest side accepted, a guard against absurd allocations (the Python decoder has none).
pub const MAX_SIDE: usize = 16385;

/// Strides of the traversal for a (2^k + 1)-node side.
pub fn strides(side: usize) -> Result<Vec<usize>> {
    let n = side.wrapping_sub(1);
    if side < 3 || !n.is_power_of_two() {
        return fail("side must be 2^k + 1");
    }
    let mut out = Vec::new();
    let mut s = S0.min(n);
    while s >= 2 {
        out.push(s);
        s /= 2;
    }
    Ok(out)
}

/// Exponent plus linear mantissa of a positive float32, read from its bits (`log2_approx`).
#[inline]
pub fn log2_approx(x: f32) -> f32 {
    let bits = x.to_bits() as i32;
    let e = ((bits >> 23) & 0xff) - 127;
    let m = (bits & 0x7f_ffff) as f32 / 8_388_608.0;
    e as f32 + m
}

const TWO_63: f32 = (1u64 << 63) as f32;

/// `np.int64(np.rint(v))` as numba runs it on x86-64: NaN and values outside i64 give i64::MIN.
#[inline]
fn rint_i64(v: f32) -> i64 {
    let r = v.round_ties_even();
    if (-TWO_63..TWO_63).contains(&r) {
        r as i64
    } else {
        i64::MIN
    }
}

/// min(max(acc + 3, 0), 6); NaN passes through, as in the Python expression.
#[inline]
fn hard_swish_gate(acc: f32) -> f32 {
    (acc + 3.0).clamp(0.0, 6.0)
}

enum Rule<'m> {
    Order0(i64),
    Ctx(f32, f32),
    Learned(&'m Model),
}

struct Predictor<'m> {
    rule: Rule<'m>,
    x: Vec<f32>,
    a1: Vec<f32>,
    a2: Vec<f32>,
}

impl Predictor<'_> {
    /// Prediction (lattice units) and table bin of node (r, c).
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn predict(
        &mut self,
        rec: &[f32],
        side: usize,
        r: usize,
        c: usize,
        s: usize,
        pass: usize,
        level: usize,
        q: i64,
    ) -> (i64, usize) {
        let h = s / 2;
        let qf = q as f32;
        let last = side as isize - 1;
        let clamp = |v: isize| v.clamp(0, last) as usize;
        let mut st = [[0f32; 3]; 4];
        for (a, row) in st.iter_mut().enumerate() {
            let o = (2 * a as isize - 3) * h as isize;
            for (b, v) in row.iter_mut().enumerate() {
                let ob = (b as isize - 1) * if pass == 0 { s } else { h } as isize;
                let (ri, ci) = if pass == 0 {
                    (clamp(r as isize + o), clamp(c as isize + ob))
                } else {
                    (clamp(r as isize + ob), clamp(c as isize + o))
                };
                *v = rec[ri * side + ci];
            }
        }
        let pos = if pass == 0 { r } else { c } as isize;
        let lo = pos - 3 * (h as isize) < 0;
        let hi = pos + 3 * (h as isize) > last;
        let p0 = if lo && hi {
            (st[1][1] + st[2][1]) / 2.0
        } else if lo {
            (3.0 * st[1][1] + 6.0 * st[2][1] - st[3][1]) / 8.0
        } else if hi {
            (6.0 * st[1][1] + 3.0 * st[2][1] - st[0][1]) / 8.0
        } else {
            (9.0 * (st[1][1] + st[2][1]) - (st[0][1] + st[3][1])) / 16.0
        };
        let mut mean = 0f32;
        for row in &st {
            for &v in row {
                mean += v;
            }
        }
        mean /= 12.0;
        let mut var = 0f32;
        for row in &st {
            for &v in row {
                let d = v - mean;
                var += d * d;
            }
        }
        let sigma = (var / 12.0).sqrt() + 0.5 * qf;
        let lsig = log2_approx(sigma / qf);
        let (p, bin) = match self.rule {
            Rule::Order0(b0) => (p0, b0),
            Rule::Ctx(a, cc) => (p0, rint_i64(a + cc * lsig)),
            Rule::Learned(m) => {
                let x = &mut self.x;
                for (k, &v) in st.iter().flatten().enumerate() {
                    x[k] = (v - p0) / sigma;
                }
                x[12] = lsig;
                for t in 0..LEVELS {
                    x[13 + t] = 0.0;
                }
                x[13 + level] = 1.0;
                x[13 + LEVELS] = pass as f32;
                x[14 + LEVELS] = log2_approx(qf * LATTICE_M_F32);
                dense(&m.layers[0].wt, &m.layers[0].b, x, &mut self.a1);
                dense(&m.layers[1].wt, &m.layers[1].b, &self.a1, &mut self.a2);
                let l3 = &m.layers[2];
                let mut o0 = l3.b[0];
                let mut o1 = l3.b[1];
                for (v, &a) in self.a2.iter().enumerate() {
                    o0 += l3.wt[2 * v] * a;
                    o1 += l3.wt[2 * v + 1] * a;
                }
                let p = p0 + sigma * o0;
                (
                    p,
                    rint_i64((o1 - LOG2_SCALE_MIN_F32) * BINS_PER_OCTAVE as f32),
                )
            }
        };
        (rint_i64(p), bin.clamp(0, BINS as i64 - 1) as usize)
    }
}

/// One hard-swish layer: out[u] = acc * gate(acc) / 6 with acc = b[u] + w[u, 0] x[0] + w[u, 1] x[1] + ...,
/// added in that order for every u. Running over v outside keeps the order and lets the outputs go in parallel.
#[inline]
fn dense(wt: &[f32], b: &[f32], x: &[f32], out: &mut [f32]) {
    let n = out.len();
    out.copy_from_slice(b);
    for (v, &xv) in x.iter().enumerate() {
        for (acc, &w) in out.iter_mut().zip(&wt[v * n..(v + 1) * n]) {
            *acc += w * xv;
        }
    }
    for acc in out.iter_mut() {
        *acc = *acc * hard_swish_gate(*acc) / 6.0;
    }
}

/// The first zstd frame of `blob`, at most `limit` bytes of it (trailing frames are ignored, as by
/// python-zstandard's `decompress`, which also requires the content size in the frame header).
fn zstd_frame(blob: &[u8], limit: usize) -> Result<Vec<u8>> {
    let has_size = blob.len() > 4 && {
        let d = blob[4];
        (d >> 6) != 0 || (d >> 5) & 1 == 1
    };
    let mut src = blob;
    let dec = match ruzstd::decoding::StreamingDecoder::new(&mut src) {
        Ok(d) => d,
        Err(e) => return fail(format!("coarse lattice is not a zstd frame: {e}")),
    };
    if !has_size {
        return fail("coarse lattice frame does not declare its content size");
    }
    let mut out = Vec::new();
    if let Err(e) = dec.take(limit as u64).read_to_end(&mut out) {
        return fail(format!("coarse lattice does not decompress: {e}"));
    }
    Ok(out)
}

/// Decodes the lattice of a multilevel product (float32 lattice values, row-major).
pub fn decode(prod: &Product<'_>, model: Option<&Model>) -> Result<Vec<f32>> {
    let side = prod.rows as usize;
    let strides = strides(side)?;
    if side > MAX_SIDE {
        return fail(format!(
            "fields larger than {MAX_SIDE} nodes per side are not supported"
        ));
    }
    let need = |kind: Kind| match prod.component(kind) {
        Some(b) => Ok(b),
        None => fail(format!("the product has no {} component", kind.name())),
    };
    let s0 = strides[0];
    let nc = (side - 1) / s0 + 1;
    let raw = zstd_frame(need(Kind::Coarse)?, nc * nc * 8 + 8)?;
    if raw.len() % 8 != 0 {
        return fail("coarse lattice is not a whole number of 8-byte values");
    }
    if raw.len() / 8 != nc * nc {
        return fail("coarse lattice has the wrong size");
    }
    let mut rec = vec![0f32; side * side];
    let mut acc = 0i64;
    for (i, chunk) in raw.chunks_exact(8).enumerate() {
        let u = u64::from_le_bytes(chunk.try_into().unwrap_or_default());
        acc = acc.wrapping_add(((u >> 1) as i64) ^ -((u & 1) as i64));
        rec[(i / nc) * s0 * side + (i % nc) * s0] = acc as f32;
    }

    let mut dec = Decoder::new(need(Kind::Stream)?, need(Kind::Raw)?)?;
    let params = need(Kind::Params)?;
    let mut off = 0usize;
    let q = 2 * prod.e as i64 + 1;
    let (n1, n2) = model.map_or((0, 0), Model::widths);
    let mut pred = Predictor {
        rule: Rule::Order0(0),
        x: vec![0f32; FEATURES],
        a1: vec![0f32; n1],
        a2: vec![0f32; n2],
    };
    for &s in &strides {
        let level = s.trailing_zeros() as usize - 1;
        let h = s / 2;
        for pass in 0..2 {
            pred.rule = match (prod.coder, model) {
                (Coder::CubicOrder0, _) => {
                    let Some(&b0) = params.get(off) else {
                        return fail("params are truncated");
                    };
                    off += 1;
                    Rule::Order0(b0 as i64)
                }
                (Coder::CubicCtx, _) => {
                    let Some(p) = params.get(off..off + 8) else {
                        return fail("params are truncated");
                    };
                    off += 8;
                    let a = f32::from_le_bytes([p[0], p[1], p[2], p[3]]);
                    let c = f32::from_le_bytes([p[4], p[5], p[6], p[7]]);
                    Rule::Ctx(a, c)
                }
                (Coder::Learned, Some(m)) => Rule::Learned(m),
                _ => return fail("this coder needs a learned model"),
            };
            let (r0, rs, c0, cs) = if pass == 0 {
                (h, s, 0, s)
            } else {
                (0, h, h, s)
            };
            for r in (r0..side).step_by(rs) {
                for c in (c0..side).step_by(cs) {
                    let (p, bin) = pred.predict(&rec, side, r, c, s, pass, level, q);
                    let k = dec.take(bin)?;
                    rec[r * side + c] = p.wrapping_add(k.wrapping_mul(q)) as f32;
                }
            }
        }
    }
    if off != params.len() || !dec.finished() {
        return fail("stream does not end where the traversal ends");
    }
    Ok(rec)
}
