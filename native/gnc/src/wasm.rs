//! WebAssembly bindings, meant to run inside a Web Worker. Timing is left to the caller.
//!
//! ```js
//! import init, { decode, describe } from "./gnc_wasm.js";
//! await init();
//! const field = decode(productBytes, modelBytes);   // modelBytes may be undefined for standalone products
//! const heights = field.heights();                   // Float32Array, metres, row-major
//! const sum = field.latticeSha256();                 // hex sha256 of the lattice as little-endian i64
//! field.free();
//! ```

use crate::{hex, tables, Decoded};
use wasm_bindgen::prelude::*;

fn js(err: crate::Error) -> JsError {
    JsError::new(err.message())
}

/// A decoded product. Heights are float32 metres rounded from the float64 values of the Python decoder; the
/// lattice (1 mm units) is exact.
#[wasm_bindgen]
pub struct DecodedField {
    inner: Decoded,
}

#[wasm_bindgen]
impl DecodedField {
    #[wasm_bindgen(getter)]
    pub fn rows(&self) -> u32 {
        self.inner.rows as u32
    }

    #[wasm_bindgen(getter)]
    pub fn cols(&self) -> u32 {
        self.inner.cols as u32
    }

    #[wasm_bindgen(getter)]
    pub fn coder(&self) -> String {
        self.inner.coder.name().to_string()
    }

    /// Bound half-width E in lattice units.
    #[wasm_bindgen(getter, js_name = boundUnits)]
    pub fn bound_units(&self) -> u32 {
        self.inner.bound_units
    }

    /// Heights in metres, row-major (a copy).
    pub fn heights(&self) -> Vec<f32> {
        self.inner.heights_f32()
    }

    /// Heights in metres as float64, identical to the Python decoder's output.
    #[wasm_bindgen(js_name = heightsF64)]
    pub fn heights_f64(&self) -> Vec<f64> {
        self.inner.heights_f64()
    }

    /// Lattice integers; throws if one does not fit 32 bits (no terrain product comes near).
    pub fn lattice(&self) -> Result<Vec<i32>, JsError> {
        self.inner
            .lattice_f32()
            .iter()
            .map(|&v| {
                let i = v as i64;
                i32::try_from(i).map_err(|_| JsError::new("lattice value beyond 32 bits"))
            })
            .collect()
    }

    /// Hex sha256 of the lattice as little-endian i64.
    #[wasm_bindgen(js_name = latticeSha256)]
    pub fn lattice_sha256(&self) -> String {
        hex(&self.inner.lattice_sha256())
    }
}

/// Decodes a product. `model` is the shared `.gnm` file a corpus product names; omit it otherwise.
#[wasm_bindgen]
pub fn decode(product: &[u8], model: Option<Vec<u8>>) -> Result<DecodedField, JsError> {
    crate::decode(product, model.as_deref())
        .map(|inner| DecodedField { inner })
        .map_err(js)
}

/// Header and byte breakdown of any product (foreign ones too) as JSON text, after every container check.
#[wasm_bindgen]
pub fn describe(product: &[u8]) -> Result<String, JsError> {
    crate::describe(product).map_err(js)
}

/// Hex id of the rANS tables compiled into this decoder.
#[wasm_bindgen(js_name = tableId)]
pub fn table_id() -> String {
    hex(&tables::TABLE_ID)
}
