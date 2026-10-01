//! The shared predictor file `.gnm` (`geoneural.codecs.predictor`): FEATURES (+ G) -> n1 -> n2 -> 2, float16
//! weights used as float32.
//!
//! ```text
//! version 1: "GNM1", 1 u16, dims 4 x u16, rANS table id (8 bytes), layers
//! version 2: "GNM1", 2 u16, dims 4 x u16, classes u16, G u16, rANS table id (8 bytes), layers, embedding
//! layers: per layer W (out x in, row-major) then b, float16 little-endian; embedding classes x G float16
//! ```
//!
//! The hash a product names is the sha256 of the version 2 serialisation, so a version 1 file is hashed after
//! rewriting its header, as the Python reader does.

use crate::tables::{FEATURES, TABLE_ID};
use crate::{fail, Result};
use sha2::{Digest, Sha256};

pub const MAGIC: &[u8; 4] = b"GNM1";

#[derive(Debug, Clone)]
pub struct Layer {
    pub inputs: usize,
    pub outputs: usize,
    /// Transposed weights, inputs x outputs (the file stores outputs x inputs), so a decoder can update all
    /// outputs of one input at a time and still add every output's terms in the file's order.
    pub wt: Vec<f32>,
    pub b: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Model {
    pub layers: [Layer; 3],
    /// classes x G, empty without an embedding.
    pub embed: Vec<f32>,
    pub embed_dims: (usize, usize),
    pub sha256: [u8; 32],
}

/// Exact float16 to float32.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    if exp == 0 {
        // Zero and subnormals: man * 2^-24, exact in float32.
        let v = man as f32 * (1.0 / 16_777_216.0);
        return if sign != 0 { -v } else { v };
    }
    if exp == 31 {
        return f32::from_bits(sign | 0x7f80_0000 | (man << 13));
    }
    f32::from_bits(sign | ((exp + 112) << 23) | (man << 13))
}

fn halfs<'a>(blob: &'a [u8], off: &mut usize, n: usize) -> Result<&'a [u8]> {
    let end = off.saturating_add(n.saturating_mul(2));
    if end > blob.len() {
        return fail("predictor file is truncated");
    }
    let out = &blob[*off..end];
    *off = end;
    Ok(out)
}

fn to_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(2)
        .map(|p| f16_to_f32(u16::from_le_bytes([p[0], p[1]])))
        .collect()
}

impl Model {
    pub fn parse(blob: &[u8]) -> Result<Model> {
        if blob.len() < 4 || &blob[..4] != MAGIC {
            return fail("not a predictor file");
        }
        let u16_at = |i: usize| -> Result<u16> {
            match blob.get(i..i + 2) {
                Some(b) => Ok(u16::from_le_bytes([b[0], b[1]])),
                None => fail("predictor file is truncated"),
            }
        };
        let version = u16_at(4)?;
        if version != 1 && version != 2 {
            return fail(format!("predictor version {version} not supported"));
        }
        let mut dims = [0usize; 4];
        for (k, d) in dims.iter_mut().enumerate() {
            *d = u16_at(6 + 2 * k)? as usize;
        }
        let (classes, g, mut off) = if version == 1 {
            (0, 0, 14)
        } else {
            (u16_at(14)? as usize, u16_at(16)? as usize, 18)
        };
        if blob.get(off..off + 8) != Some(&TABLE_ID[..]) {
            return fail("predictor was trained for other rANS tables");
        }
        off += 8;
        let body_start = off;
        let mut layer = |k: usize| -> Result<Layer> {
            let (a, b) = (dims[k], dims[k + 1]);
            let w = to_f32(halfs(blob, &mut off, a * b)?);
            let wt = (0..a * b).map(|i| w[(i % b) * a + i / b]).collect();
            let bias = to_f32(halfs(blob, &mut off, b)?);
            Ok(Layer {
                inputs: a,
                outputs: b,
                wt,
                b: bias,
            })
        };
        let layers = [layer(0)?, layer(1)?, layer(2)?];
        let mut embed = Vec::new();
        if classes > 0 {
            embed = to_f32(halfs(blob, &mut off, classes * g)?);
        }
        if off != blob.len() {
            return fail("predictor file has trailing or missing bytes");
        }
        // An embedding of size zero counts as none, as in the Python class.
        let embed_dims = if embed.is_empty() {
            (0, 0)
        } else {
            (classes, g)
        };
        if dims[0] != FEATURES + embed_dims.1 || dims[3] != 2 {
            return fail("predictor must be FEATURES (+ embedding) -> n1 -> n2 -> 2");
        }
        let mut h = Sha256::new();
        h.update(MAGIC);
        h.update(2u16.to_le_bytes());
        for d in dims {
            h.update((d as u16).to_le_bytes());
        }
        h.update((embed_dims.0 as u16).to_le_bytes());
        h.update((embed_dims.1 as u16).to_le_bytes());
        h.update(TABLE_ID);
        h.update(&blob[body_start..]);
        Ok(Model {
            layers,
            embed,
            embed_dims,
            sha256: h.finalize().into(),
        })
    }

    pub fn classes(&self) -> usize {
        self.embed_dims.0
    }

    pub fn widths(&self) -> (usize, usize) {
        (self.layers[0].outputs, self.layers[1].outputs)
    }
}
