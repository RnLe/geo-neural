//! Reading the fixtures written by `geoneural.physics.fixtures`.
#![allow(dead_code)]

use landscape_core::{
    Activation, Apply, Boundary, Closure, Grid, LayerSpec, Model, Params, Scenario, Teacher,
    Validated,
};
use serde_json::Value;
use std::path::PathBuf;

pub fn path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join(name)
}

pub fn load(case: &str) -> Value {
    let text = std::fs::read_to_string(path(&format!("{case}.json")))
        .expect("fixture missing; run the exporter");
    serde_json::from_str(&text).unwrap()
}

fn decode(text: &str) -> Vec<u8> {
    let value = |c: u8| -> u32 {
        match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a' + 26) as u32,
            b'0'..=b'9' => (c - b'0' + 52) as u32,
            b'+' => 62,
            b'/' => 63,
            _ => panic!("bad base64 byte {c}"),
        }
    };
    let digits: Vec<u8> = text.bytes().filter(|&c| c != b'=').collect();
    let mut out = Vec::with_capacity(digits.len() * 3 / 4);
    for chunk in digits.chunks(4) {
        let acc = chunk
            .iter()
            .enumerate()
            .fold(0, |acc, (i, &c)| acc | value(c) << (18 - 6 * i));
        for i in 0..chunk.len() - 1 {
            out.push((acc >> (16 - 8 * i)) as u8);
        }
    }
    out
}

pub fn f64s(array: &Value) -> Vec<f64> {
    assert_eq!(array["dtype"], "f64");
    decode(array["base64"].as_str().unwrap())
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

pub fn f32s(array: &Value) -> Vec<f32> {
    assert_eq!(array["dtype"], "f32");
    decode(array["base64"].as_str().unwrap())
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

pub fn num(case: &Value, key: &str) -> f64 {
    case[key]
        .as_f64()
        .unwrap_or_else(|| panic!("missing {key}"))
}

/// One arm of `closure.json` with its weights.
pub fn closure(arm: &str) -> Closure {
    let meta: Value =
        serde_json::from_str(&std::fs::read_to_string(path("closure.json")).unwrap()).unwrap();
    let spec = &meta["arms"][arm];
    let bytes = std::fs::read(path(spec["file"].as_str().unwrap())).unwrap();
    let weights: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(weights.len() as u64, spec["floats"].as_u64().unwrap());
    let index = |v: &Value, k: &str| v[k].as_u64().unwrap() as usize;
    let layers: Vec<LayerSpec> = spec["layers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| LayerSpec {
            cin: index(l, "in"),
            cout: index(l, "out"),
            weight_offset: index(l, "weightOffset"),
            bias_offset: index(l, "biasOffset"),
            activation: Activation::parse(l["activation"].as_str().unwrap()).unwrap(),
        })
        .collect();
    let v = if spec["validated"].is_object() {
        &spec["validated"]
    } else {
        &meta["validated"]
    };
    let apply = match spec["apply"].as_str().unwrap() {
        "conductance" => Apply::conductance(num(spec, "floor"), num(spec, "aMax")).unwrap(),
        name => Apply::parse(name).unwrap(),
    };
    Closure::from_f32(
        apply,
        &layers,
        &weights,
        num(&meta, "spacingM"),
        Teacher {
            diffusivity: num(&meta["teacher"], "diffusivity"),
            critical_slope: num(&meta["teacher"], "criticalSlope"),
        },
        Validated {
            max_slope: num(v, "maxSlope"),
            min_height_m: num(v, "minHeightM"),
            max_height_m: num(v, "maxHeightM"),
        },
    )
    .unwrap()
}

/// A scenario built from a fixture's grid, boundary and initial surface.
pub fn scenario(case: &Value, model: Model) -> Scenario {
    let grid = Grid::new(
        case["side"].as_u64().unwrap() as usize,
        num(case, "spacingM"),
    )
    .unwrap();
    let boundary = Boundary::parse(case["boundary"].as_str().unwrap()).unwrap();
    let params = Params::defaults(&model);
    Scenario::new(grid, boundary, model, params, &f64s(&case["initial"])).unwrap()
}

/// `max |a - b| / max |b|`.
pub fn relative(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let diff = a
        .iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
    diff / b.iter().fold(0.0f64, |m, y| m.max(y.abs()))
}

/// `sum |h| * a`: the scale conservation residuals are measured against.
pub fn mass_scale(height: &[f64], spacing_m: f64) -> f64 {
    height.iter().map(|h| h.abs()).sum::<f64>() * spacing_m * spacing_m
}
