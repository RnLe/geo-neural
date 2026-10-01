//! Exact decoder of `.gnc` terrain products written by `geoneural.codecs.package`.
//!
//! Supports the multilevel coders `cubic-order0`, `cubic-ctx` and `learned` with a uniform bound. Products with a
//! geology context or a bound-allocation rule, models with a class embedding and foreign (conventional) streams
//! are refused with an error. Every check of the Python reader is repeated: magic, version, coder, node
//! centring, rANS table id, directory and payload lengths, crc32 of every component and the model's sha256.
//!
//! The decoded field is bit-identical to `geoneural.codecs.package.decode`: the lattice values are the same
//! float32 numbers, and [`Decoded::heights_f64`] gives the same float64 metres.
//!
//! ```no_run
//! let product = std::fs::read("essen.gnc").unwrap();
//! let model = std::fs::read("shared.gnm").unwrap();
//! let field = gnc::decode(&product, Some(&model)).unwrap();
//! println!("{} x {}, first height {} m", field.rows, field.cols, field.heights_f64()[0]);
//! ```

pub mod container;
pub mod model;
pub mod multilevel;
pub mod rans;
#[rustfmt::skip]
pub mod tables;
#[cfg(target_arch = "wasm32")]
mod wasm;

use std::fmt;

pub use container::{Coder, Kind, Product};
pub use model::Model;

/// A decoding failure, with a message that says which check failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(String);

impl Error {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Error(message.into())
    }

    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn fail<T>(message: impl Into<String>) -> Result<T> {
    Err(Error::new(message))
}

/// Lowercase hex of a byte string.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A decoded field: row-major lattice values (units of [`tables::LATTICE_M`]) as the Python decoder holds them.
#[derive(Debug, Clone)]
pub struct Decoded {
    pub rows: usize,
    pub cols: usize,
    pub coder: Coder,
    /// Bound half-width in lattice units (0: lossless on the lattice).
    pub bound_units: u32,
    values: Vec<f32>,
}

impl Decoded {
    /// Lattice values as float32 (integers; exact below 2^24 in magnitude).
    pub fn lattice_f32(&self) -> &[f32] {
        &self.values
    }

    /// Lattice values as integers.
    pub fn lattice(&self) -> Vec<i64> {
        self.values.iter().map(|&v| v as i64).collect()
    }

    /// Heights in metres, float64: `lattice * LATTICE_M`, exactly what `package.decode` returns.
    pub fn heights_f64(&self) -> Vec<f64> {
        self.values
            .iter()
            .map(|&v| v as f64 * tables::LATTICE_M)
            .collect()
    }

    /// Heights in metres rounded to float32 (from the float64 values).
    pub fn heights_f32(&self) -> Vec<f32> {
        self.values
            .iter()
            .map(|&v| (v as f64 * tables::LATTICE_M) as f32)
            .collect()
    }

    /// sha256 of the lattice as little-endian i64, the checksum the fixtures and the web bundle record.
    pub fn lattice_sha256(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        for chunk in self.values.chunks(4096) {
            let bytes: Vec<u8> = chunk
                .iter()
                .flat_map(|&v| (v as i64).to_le_bytes())
                .collect();
            h.update(&bytes);
        }
        h.finalize().into()
    }
}

/// Decodes a product from its bytes. `model` is the shared predictor file (`.gnm`) a corpus product names in
/// its header; it is ignored when the model is embedded and by the fixed coders.
pub fn decode(product: &[u8], model: Option<&[u8]>) -> Result<Decoded> {
    let prod = container::read(product)?;
    decode_product(&prod, model)
}

/// Decodes a product that was already read with [`container::read`].
pub fn decode_product(prod: &Product<'_>, model: Option<&[u8]>) -> Result<Decoded> {
    if prod.coder == Coder::Foreign {
        return fail("foreign products are decoded by their own codec");
    }
    if prod.component(Kind::Context).is_some() {
        return fail("this product carries a geology context component, which this decoder does not implement");
    }
    if prod.component(Kind::Rule).is_some() {
        return fail("this product carries a bound-allocation rule component, which this decoder does not implement");
    }
    if prod.rows != prod.cols {
        return fail("square products only");
    }
    let parsed;
    let predictor = if prod.coder == Coder::Learned {
        let blob = if prod.model_embedded {
            match prod.component(Kind::Model) {
                Some(b) => b,
                None => {
                    return fail(
                        "the product says its model is embedded but has no model component",
                    )
                }
            }
        } else {
            match model {
                Some(b) => b,
                None => {
                    let sha = prod.model_sha256.map(|s| hex(&s)).unwrap_or_default();
                    return fail(format!(
                        "this product needs the shared model {}",
                        &sha[..sha.len().min(12)]
                    ));
                }
            }
        };
        parsed = Model::parse(blob)?;
        if prod.model_sha256 != Some(parsed.sha256) {
            return fail("the given model is not the one this product was encoded with");
        }
        if parsed.classes() > 0 {
            return fail("this product's model has a class embedding and needs a geology context, which this decoder does not implement");
        }
        Some(&parsed)
    } else {
        None
    };
    let values = multilevel::decode(prod, predictor)?;
    Ok(Decoded {
        rows: prod.rows as usize,
        cols: prod.cols as usize,
        coder: prod.coder,
        bound_units: prod.e,
        values,
    })
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_num(v: f64) -> String {
    if v.is_finite() {
        format!("{v}")
    } else {
        "null".to_string()
    }
}

/// Header and byte breakdown of any product (foreign ones too) as JSON text, after every container check.
pub fn describe(product: &[u8]) -> Result<String> {
    let p = container::read(product)?;
    let parts: Vec<String> = p
        .breakdown()
        .iter()
        .map(|(k, n)| format!("[{},{n}]", json_str(k)))
        .collect();
    Ok(format!(
        "{{\"coder\":{},\"foreign\":{},\"rows\":{},\"cols\":{},\"boundUnits\":{},\"latticeM\":{},\"west\":{},\
         \"north\":{},\"spacingM\":{},\"epsgH\":{},\"epsgV\":{},\"modelSha256\":{},\"modelEmbedded\":{},\
         \"bytes\":{},\"breakdown\":[{}]}}",
        json_str(p.coder.name()),
        json_str(&p.foreign),
        p.rows,
        p.cols,
        p.e,
        json_num(p.lattice_m),
        json_num(p.west),
        json_num(p.north),
        json_num(p.spacing_m),
        p.epsg_h,
        p.epsg_v,
        p.model_sha256.map_or("null".to_string(), |s| json_str(&hex(&s))),
        p.model_embedded,
        product.len(),
        parts.join(",")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn table_id_matches_the_frequencies() {
        let mut h = Sha256::new();
        for row in tables::FREQ.iter() {
            for &f in row {
                h.update((f as i32).to_le_bytes());
            }
        }
        h.update([tables::PROB_BITS as u8, tables::BINS_PER_OCTAVE]);
        assert_eq!(h.finalize()[..8], tables::TABLE_ID);
        for (freq, cdf) in tables::FREQ.iter().zip(tables::CDF.iter()) {
            assert_eq!(cdf[0], 0);
            for t in 0..tables::TOKENS {
                assert!(freq[t] >= 1);
                assert_eq!(cdf[t + 1] as u32, cdf[t] as u32 + freq[t] as u32);
            }
            assert_eq!(cdf[tables::TOKENS] as u32, 1 << tables::PROB_BITS);
        }
    }

    #[test]
    fn float16_converts_exactly() {
        use model::f16_to_f32;
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x8000).to_bits(), 0x8000_0000);
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
    }

    #[test]
    fn log2_approx_reads_the_bits() {
        use multilevel::log2_approx;
        assert_eq!(log2_approx(1.0), 0.0);
        assert_eq!(log2_approx(8.0), 3.0);
        assert_eq!(log2_approx(0.75), -0.5);
        assert_eq!(log2_approx(3.0), 1.5);
    }

    #[test]
    fn strides_follow_the_side() {
        assert_eq!(
            multilevel::strides(1025).unwrap(),
            vec![64, 32, 16, 8, 4, 2]
        );
        assert_eq!(multilevel::strides(9).unwrap(), vec![8, 4, 2]);
        assert_eq!(multilevel::strides(3).unwrap(), vec![2]);
        for bad in [0, 1, 2, 4, 1024] {
            assert!(multilevel::strides(bad).is_err());
        }
    }
}
