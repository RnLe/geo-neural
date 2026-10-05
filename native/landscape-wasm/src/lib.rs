//! WebAssembly bindings for `landscape-core`: one `Scenario` class, meant to
//! run inside a Web Worker.
//!
//! ```js
//! import init, { Scenario } from "./pkg/landscape_wasm.js";
//! await init();
//! const s = new Scenario(48, 50, heights, "nonlinear", "closed", '{"diffusivity":0.05}');
//! const d = s.step(10, 200);   // plain object, see `step`
//! const h = s.surface();       // Float32Array copy
//! s.free();
//! ```

use js_sys::{Array, Object, Reflect, JSON};
use landscape_core::{
    Activation, Apply, Boundary, Closure, Diagnostics, Grid, LayerSpec, Model, Params, Teacher,
    Validated,
};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(typescript_custom_section)]
const DIAGNOSTICS: &str = r#"
/** Returned by `Scenario.step` and `Scenario.diagnostics`. Volumes in m^3, cumulative since reset. */
export interface Diagnostics {
    timeYears: number;
    advancedYears: number;
    stepsDone: number;
    substeps: number;
    maxSubstepYears: number;
    stableDtYears: number;
    integralM3: number;
    initialIntegralM3: number;
    /** Material that entered through the boundary; negative when it left. */
    boundaryExchangeM3: number;
    sourcesM3: number;
    /** integral - initialIntegral - boundaryExchange - sources. */
    residualM3: number;
    residualRelative: number;
    maxSlope: number;
    minHeightM: number;
    maxHeightM: number;
    limitedFaces: number;
    rejections: number;
    rejected: boolean;
    truncated: boolean;
    message: string;
}
"#;

/// A surface evolving under one model.
#[wasm_bindgen]
pub struct Scenario {
    inner: landscape_core::Scenario,
}

#[wasm_bindgen]
impl Scenario {
    /// - `side`, `spacing_m`: a `side * side` grid of cells, metres.
    /// - `initial`: `side * side` heights in metres, row-major, row 0 north.
    /// - `model`: `"linear"`, `"nonlinear"`, `"flux"`, `"kfield"`, `"penalty"` or `"conductance"`.
    /// - `boundary`: `"closed"`, `"fixed"` or `"periodic"` (learned models: closed or fixed).
    /// - `params_json`: `diffusivity` (m^2/yr, default 0.05) for linear and
    ///   nonlinear; `criticalSlope` (default 0.6) for nonlinear; `uplift`
    ///   (m/yr, default 0), `safety` (default 0.9) and `maxSubsteps` (default
    ///   512, or 4 for learned models) for all. Other keys are refused.
    /// - `weights`, `weights_meta_json`: for the learned models, the arm's
    ///   float32 weights and the full `closure.json` text.
    /// - `diffusivity`: optional per-cell diffusivity for `"linear"`, m^2/yr.
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        side: usize,
        spacing_m: f64,
        initial: &[f64],
        model: &str,
        boundary: &str,
        params_json: &str,
        weights: Option<Vec<f32>>,
        weights_meta_json: Option<String>,
        diffusivity: Option<Vec<f64>>,
    ) -> Result<Scenario, JsError> {
        let grid = Grid::new(side, spacing_m)?;
        let boundary = Boundary::parse(boundary)?;
        let params = parse(if params_json.trim().is_empty() {
            "{}"
        } else {
            params_json
        })?;
        let learned = matches!(model, "flux" | "kfield" | "penalty" | "conductance");
        let allowed: &[&str] = match model {
            "linear" => &["diffusivity", "uplift", "safety", "maxSubsteps"],
            "nonlinear" => &[
                "diffusivity",
                "criticalSlope",
                "uplift",
                "safety",
                "maxSubsteps",
            ],
            _ => &["uplift", "safety", "maxSubsteps"],
        };
        for key in Object::keys(params.unchecked_ref::<Object>()).iter() {
            let key = key.as_string().unwrap_or_default();
            if !allowed.contains(&key.as_str()) {
                return Err(JsError::new(&format!(
                    "parameter {key:?} does not apply to model {model:?}"
                )));
            }
        }
        if diffusivity.is_some() && model != "linear" {
            return Err(JsError::new(
                "a diffusivity field applies only to the linear model",
            ));
        }
        let d = number(&params, "diffusivity")?.unwrap_or(0.05);
        let model = match model {
            "linear" => Model::Linear {
                diffusivity: d,
                field: diffusivity,
            },
            "nonlinear" => Model::Nonlinear(Teacher {
                diffusivity: d,
                critical_slope: number(&params, "criticalSlope")?.unwrap_or(0.6),
            }),
            arm if learned => {
                let weights = weights.ok_or_else(|| JsError::new("learned models need weights"))?;
                let meta = weights_meta_json
                    .ok_or_else(|| JsError::new("learned models need the weights meta"))?;
                Model::Learned(closure(arm, &weights, &parse(&meta)?)?)
            }
            other => return Err(JsError::new(&format!("unknown model {other:?}"))),
        };
        let mut stepping = Params::defaults(&model);
        stepping.uplift_m_per_year = number(&params, "uplift")?.unwrap_or(0.0);
        stepping.safety = number(&params, "safety")?.unwrap_or(stepping.safety);
        if let Some(cap) = number(&params, "maxSubsteps")? {
            if !(cap >= 1.0 && cap <= f64::from(u32::MAX) && cap.fract() == 0.0) {
                return Err(JsError::new("maxSubsteps must be a positive integer"));
            }
            stepping.max_substeps = cap as u32;
        }
        let inner = landscape_core::Scenario::new(grid, boundary, model, stepping, initial)?;
        Ok(Scenario { inner })
    }

    /// Advances `steps` steps of `dt_years`, splitting each into stable
    /// substeps, and returns a plain object:
    ///
    /// `timeYears, advancedYears, stepsDone, substeps, maxSubstepYears,
    /// stableDtYears, integralM3, initialIntegralM3, boundaryExchangeM3,
    /// sourcesM3, residualM3, residualRelative, maxSlope, minHeightM,
    /// maxHeightM, limitedFaces, rejections, rejected, truncated, message`.
    ///
    /// A call stops after `maxSubsteps` substeps (`truncated`, check
    /// `advancedYears`) or before a state the model refuses (`rejected`, with
    /// `message`).
    #[wasm_bindgen(unchecked_return_type = "Diagnostics")]
    pub fn step(&mut self, steps: u32, dt_years: f64) -> JsValue {
        to_object(&self.inner.step(steps, dt_years))
    }

    /// The same object as `step` returns, for the current state, without stepping.
    #[wasm_bindgen(unchecked_return_type = "Diagnostics")]
    pub fn diagnostics(&self) -> JsValue {
        to_object(&self.inner.diagnostics())
    }

    /// A Float32Array copy of the current heights.
    pub fn surface(&self) -> Vec<f32> {
        self.inner.height().iter().map(|&h| h as f32).collect()
    }

    /// Replaces the surface and clears time and the ledger.
    pub fn reset(&mut self, initial: &[f64]) -> Result<(), JsError> {
        Ok(self.inner.reset(initial)?)
    }
}

fn parse(text: &str) -> Result<JsValue, JsError> {
    let value = JSON::parse(text).map_err(|_| JsError::new("invalid JSON"))?;
    if value.is_object() {
        Ok(value)
    } else {
        Err(JsError::new("expected a JSON object"))
    }
}

fn field(object: &JsValue, key: &str) -> Result<JsValue, JsError> {
    if !object.is_object() {
        return Err(JsError::new(&format!(
            "cannot read {key:?} from a non-object"
        )));
    }
    Reflect::get(object, &JsValue::from_str(key))
        .map_err(|_| JsError::new(&format!("cannot read {key:?}")))
}

fn number(object: &JsValue, key: &str) -> Result<Option<f64>, JsError> {
    let value = field(object, key)?;
    if value.is_undefined() || value.is_null() {
        return Ok(None);
    }
    value
        .as_f64()
        .map(Some)
        .ok_or_else(|| JsError::new(&format!("{key:?} must be a number")))
}

fn required(object: &JsValue, key: &str) -> Result<f64, JsError> {
    number(object, key)?.ok_or_else(|| JsError::new(&format!("missing {key:?}")))
}

fn text(object: &JsValue, key: &str) -> Result<String, JsError> {
    field(object, key)?
        .as_string()
        .ok_or_else(|| JsError::new(&format!("{key:?} must be a string")))
}

fn index(object: &JsValue, key: &str) -> Result<usize, JsError> {
    let value = required(object, key)?;
    if value >= 0.0 && value.fract() == 0.0 && value < 1e15 {
        Ok(value as usize)
    } else {
        Err(JsError::new(&format!(
            "{key:?} must be a non-negative integer"
        )))
    }
}

/// Reads one arm of `closure.json` (see `geoneural.physics.fixtures`).
fn closure(arm: &str, weights: &[f32], meta: &JsValue) -> Result<Closure, JsError> {
    let spec = field(&field(meta, "arms")?, arm)?;
    if spec.is_undefined() {
        return Err(JsError::new(&format!(
            "the weights meta has no arm {arm:?}"
        )));
    }
    let floats = index(&spec, "floats")?;
    if weights.len() != floats {
        return Err(JsError::new(&format!(
            "arm {arm:?} needs {floats} weights, got {}",
            weights.len()
        )));
    }
    let layers: Array = field(&spec, "layers")?
        .dyn_into()
        .map_err(|_| JsError::new("\"layers\" must be an array"))?;
    let mut specs = Vec::with_capacity(layers.length() as usize);
    for layer in layers.iter() {
        specs.push(LayerSpec {
            cin: index(&layer, "in")?,
            cout: index(&layer, "out")?,
            weight_offset: index(&layer, "weightOffset")?,
            bias_offset: index(&layer, "biasOffset")?,
            activation: Activation::parse(&text(&layer, "activation")?)?,
        });
    }
    let teacher = field(meta, "teacher")?;
    // An arm trained on its own seed carries its own range; the others share the file's.
    let own = field(&spec, "validated")?;
    let validated = if own.is_undefined() {
        field(meta, "validated")?
    } else {
        own
    };
    let apply = match text(&spec, "apply")?.as_str() {
        "conductance" => Apply::conductance(required(&spec, "floor")?, required(&spec, "aMax")?)?,
        name => Apply::parse(name)?,
    };
    Ok(Closure::from_f32(
        apply,
        &specs,
        weights,
        required(meta, "spacingM")?,
        Teacher {
            diffusivity: required(&teacher, "diffusivity")?,
            critical_slope: required(&teacher, "criticalSlope")?,
        },
        Validated {
            max_slope: required(&validated, "maxSlope")?,
            min_height_m: required(&validated, "minHeightM")?,
            max_height_m: required(&validated, "maxHeightM")?,
        },
    )?)
}

fn to_object(d: &Diagnostics) -> JsValue {
    let object = Object::new();
    let entries: [(&str, JsValue); 20] = [
        ("timeYears", d.time_years.into()),
        ("advancedYears", d.advanced_years.into()),
        ("stepsDone", d.steps_done.into()),
        ("substeps", d.substeps.into()),
        ("maxSubstepYears", d.max_substep_years.into()),
        ("stableDtYears", d.stable_dt_years.into()),
        ("integralM3", d.integral_m3.into()),
        ("initialIntegralM3", d.initial_integral_m3.into()),
        ("boundaryExchangeM3", d.boundary_exchange_m3.into()),
        ("sourcesM3", d.sources_m3.into()),
        ("residualM3", d.residual_m3.into()),
        ("residualRelative", d.residual_relative.into()),
        ("maxSlope", d.max_slope.into()),
        ("minHeightM", d.min_height_m.into()),
        ("maxHeightM", d.max_height_m.into()),
        ("limitedFaces", (d.limited_faces as f64).into()),
        ("rejections", d.rejections.into()),
        ("rejected", d.rejected.into()),
        ("truncated", d.truncated.into()),
        ("message", d.message.as_str().into()),
    ];
    for (key, value) in entries {
        // Setting a plain property on a fresh object cannot fail.
        let _ = Reflect::set(&object, &JsValue::from_str(key), &value);
    }
    object.into()
}
