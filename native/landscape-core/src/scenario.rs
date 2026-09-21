//! A running scenario: state, model, boundary, substepping and the ledger.

use crate::closure::{min_max, Closure, Work};
use crate::grid::{face_gradients, max_abs, Boundary, Grid};
use crate::physics::{
    linear_stable_dt, linear_tendency, teacher_stable_dt, teacher_tendency, variable_tendency,
    Teacher,
};
use crate::{fail, Error};

/// What drives the surface.
#[derive(Clone, Debug)]
pub enum Model {
    /// Linear diffusion with a uniform `diffusivity`, or with a per-cell
    /// `field` (m^2/yr, one value per cell) that replaces it.
    Linear {
        diffusivity: f64,
        field: Option<Vec<f64>>,
    },
    /// The critical-slope teacher.
    Nonlinear(Teacher),
    /// A learned closure. Closed or fixed boundaries only.
    Learned(Closure),
}

/// Stepping controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Uniform uplift on every free cell, m/yr. Booked as a source.
    pub uplift_m_per_year: f64,
    /// Each substep is at most `safety` times the stability bound, in (0, 1].
    pub safety: f64,
    /// Most substeps one [`Scenario::step`] call may take, so a call stays bounded.
    pub max_substeps: u32,
}

impl Params {
    /// No uplift, safety 0.9, and at most 512 substeps per call, or 4 for a
    /// learned closure, whose substeps cost far more.
    pub fn defaults(model: &Model) -> Self {
        let max_substeps = if matches!(model, Model::Learned(_)) {
            4
        } else {
            512
        };
        Self {
            uplift_m_per_year: 0.0,
            safety: 0.9,
            max_substeps,
        }
    }
}

/// What a call to [`Scenario::step`] did, and the state of the ledger after it.
///
/// Volumes are in m^3 and cumulative since construction or the last reset.
/// For a conservative model `residual_m3` is rounding; for the K-field arms it
/// is the material the operator created or destroyed.
#[derive(Clone, Debug, PartialEq)]
pub struct Diagnostics {
    /// Physical time since the last reset.
    pub time_years: f64,
    /// Physical time advanced by this call. Authoritative when `truncated`.
    pub advanced_years: f64,
    /// Requested steps completed by this call.
    pub steps_done: u32,
    /// Substeps taken by this call.
    pub substeps: u32,
    /// Longest substep of this call.
    pub max_substep_years: f64,
    /// Stability bound at the last substep, before `safety`. NaN if never evaluated.
    pub stable_dt_years: f64,
    /// `sum h_i * a` now.
    pub integral_m3: f64,
    /// `sum h_i * a` at the last reset.
    pub initial_integral_m3: f64,
    /// Material that entered through the boundary; negative when it left.
    pub boundary_exchange_m3: f64,
    /// Material added by uplift.
    pub sources_m3: f64,
    /// `integral - initial_integral - boundary_exchange - sources`.
    pub residual_m3: f64,
    /// `residual_m3 / sum |h_i| a`, or 0 for an all-zero surface.
    pub residual_relative: f64,
    /// Largest `|dh/dx|` over all faces.
    pub max_slope: f64,
    pub min_height_m: f64,
    pub max_height_m: f64,
    /// Faces where the teacher clipped `|g| / S_c` to 0.99, summed over substeps.
    pub limited_faces: u64,
    /// Calls that were refused, since the last reset.
    pub rejections: u32,
    /// This call was refused; the state is as it was before the refused substep.
    pub rejected: bool,
    /// This call stopped at `max_substeps` before finishing.
    pub truncated: bool,
    /// Why the call was refused, or empty.
    pub message: String,
}

#[derive(Default)]
struct Call {
    advanced: f64,
    steps_done: u32,
    substeps: u32,
    max_substep: f64,
    rejected: bool,
    truncated: bool,
    message: String,
}

/// A surface evolving under one model and one boundary mode.
#[derive(Clone, Debug)]
pub struct Scenario {
    grid: Grid,
    boundary: Boundary,
    model: Model,
    params: Params,
    height: Vec<f64>,
    tendency: Vec<f64>,
    east: Vec<f64>,
    south: Vec<f64>,
    work: Work,
    time_years: f64,
    initial_integral: f64,
    exchange_m3: f64,
    sources_m3: f64,
    limited_faces: u64,
    rejections: u32,
    last_bound: f64,
}

impl Scenario {
    pub fn new(
        grid: Grid,
        boundary: Boundary,
        model: Model,
        params: Params,
        initial: &[f64],
    ) -> Result<Self, Error> {
        // Written so that NaN fails every check.
        let non_negative = |v: f64| v.is_finite() && v >= 0.0;
        let positive = |v: f64| v.is_finite() && v > 0.0;
        match &model {
            Model::Linear { diffusivity, field } => {
                if !non_negative(*diffusivity) {
                    return fail("diffusivity must be finite and non-negative");
                }
                if let Some(k) = field {
                    if k.len() != grid.cells() || !k.iter().all(|&v| non_negative(v)) {
                        return fail(
                            "the diffusivity field needs side * side finite non-negative values",
                        );
                    }
                }
            }
            Model::Nonlinear(t) => {
                if !(non_negative(t.diffusivity) && positive(t.critical_slope)) {
                    return fail("the teacher needs a non-negative diffusivity and a positive critical slope");
                }
            }
            Model::Learned(c) => {
                if boundary.periodic() {
                    return fail(
                        "learned closures were trained with closed boundaries; use closed or fixed",
                    );
                }
                if (grid.spacing_m - c.spacing_m).abs() > 1e-9 * c.spacing_m {
                    return fail(format!(
                        "this closure was trained at {} m spacing and is only valid there",
                        c.spacing_m
                    ));
                }
            }
        }
        let safety_ok = positive(params.safety) && params.safety <= 1.0;
        if !(params.uplift_m_per_year.is_finite() && safety_ok) {
            return fail("uplift must be finite and safety in (0, 1]");
        }
        if params.max_substeps == 0 {
            return fail("max_substeps must be at least 1");
        }
        let mut scenario = Self {
            grid,
            boundary,
            model,
            params,
            height: Vec::new(),
            tendency: vec![0.0; grid.cells()],
            east: Vec::new(),
            south: Vec::new(),
            work: Work::default(),
            time_years: 0.0,
            initial_integral: 0.0,
            exchange_m3: 0.0,
            sources_m3: 0.0,
            limited_faces: 0,
            rejections: 0,
            last_bound: f64::NAN,
        };
        scenario.reset(initial)?;
        Ok(scenario)
    }

    /// Replaces the surface and clears time and the ledger. Under a fixed
    /// boundary the new outer ring becomes the base level.
    pub fn reset(&mut self, initial: &[f64]) -> Result<(), Error> {
        if initial.len() != self.grid.cells() {
            return fail(format!(
                "expected {} heights, got {}",
                self.grid.cells(),
                initial.len()
            ));
        }
        if !initial.iter().all(|v| v.is_finite()) {
            return fail("heights must be finite");
        }
        self.height = initial.to_vec();
        self.time_years = 0.0;
        self.initial_integral = self.grid.integral(initial);
        self.exchange_m3 = 0.0;
        self.sources_m3 = 0.0;
        self.limited_faces = 0;
        self.rejections = 0;
        self.last_bound = self.tendency().map_or(f64::NAN, |(bound, _)| bound);
        Ok(())
    }

    pub fn grid(&self) -> Grid {
        self.grid
    }

    pub fn boundary(&self) -> Boundary {
        self.boundary
    }

    pub fn params(&self) -> Params {
        self.params
    }

    pub fn height(&self) -> &[f64] {
        &self.height
    }

    /// Advances `steps` steps of `dt_years` each.
    ///
    /// Each step is split into equal substeps no longer than `safety` times
    /// the stability bound, re-evaluated at every substep. A step within the
    /// bound is one plain Euler update `h += dt * tendency`. The call stops
    /// early after `max_substeps` substeps (`truncated`), or before a substep
    /// whose state the model refuses (`rejected`).
    pub fn step(&mut self, steps: u32, dt_years: f64) -> Diagnostics {
        let mut call = Call::default();
        if !(dt_years.is_finite() && dt_years >= 0.0) {
            self.reject(
                &mut call,
                format!("dt_years must be finite and non-negative, got {dt_years}"),
            );
            return self.report(&call);
        }
        'steps: for _ in 0..steps {
            let mut remaining = dt_years;
            while remaining > 0.0 {
                if call.substeps == self.params.max_substeps {
                    call.truncated = true;
                    break 'steps;
                }
                let (bound, clipped) = match self.tendency() {
                    Ok(found) => found,
                    Err(message) => {
                        self.reject(&mut call, message);
                        break 'steps;
                    }
                };
                self.limited_faces += clipped as u64;
                self.last_bound = bound;
                let pieces = (remaining / (self.params.safety * bound)).ceil();
                let dt = if pieces > 1.0 {
                    remaining / pieces
                } else {
                    remaining
                };
                self.advance(dt);
                remaining -= dt;
                call.advanced += dt;
                call.substeps += 1;
                call.max_substep = call.max_substep.max(dt);
            }
            call.steps_done += 1;
        }
        self.report(&call)
    }

    /// The ledger and state now, without stepping.
    pub fn diagnostics(&self) -> Diagnostics {
        self.report(&Call::default())
    }

    fn reject(&mut self, call: &mut Call, message: String) {
        self.rejections += 1;
        call.rejected = true;
        call.message = message;
    }

    /// Fills `self.tendency` at the current state. Returns the stability
    /// bound and the number of clipped teacher faces, or why the state is refused.
    fn tendency(&mut self) -> Result<(f64, usize), String> {
        let Self {
            grid,
            boundary,
            model,
            height,
            tendency,
            east,
            south,
            work,
            ..
        } = self;
        let periodic = boundary.periodic();
        let dx = grid.spacing_m;
        let mut clipped = 0;
        let bound = match model {
            Model::Linear {
                diffusivity,
                field: None,
            } => {
                linear_tendency(grid, periodic, height, *diffusivity, east, south, tendency);
                linear_stable_dt(dx, *diffusivity)
            }
            Model::Linear { field: Some(k), .. } => {
                variable_tendency(grid, periodic, height, k, east, south, tendency);
                linear_stable_dt(dx, k.iter().fold(0.0, |m, &v| m.max(v)))
            }
            Model::Nonlinear(teacher) => {
                clipped = teacher_tendency(grid, periodic, height, teacher, east, south, tendency);
                teacher_stable_dt(dx, teacher)
            }
            Model::Learned(closure) => {
                face_gradients(grid, false, height, east, south);
                closure.check(height, max_abs(east).max(max_abs(south)))?;
                match closure.tendency(grid, height, work, tendency) {
                    // The K-field update is a convex combination below dx^2 / (4 K_max).
                    Some(k_max) => linear_stable_dt(dx, k_max),
                    // A learned flux has no closed bound; use the teacher it imitates.
                    None => teacher_stable_dt(dx, &closure.teacher),
                }
            }
        };
        if tendency.iter().all(|v| v.is_finite()) {
            Ok((bound, clipped))
        } else {
            Err("the tendency is not finite".into())
        }
    }

    /// One Euler update with the stored tendency. Under a fixed boundary the
    /// outer ring is left alone and what the ring would have gained is booked
    /// as material leaving the domain.
    fn advance(&mut self, dt: f64) {
        let fixed = self.boundary == Boundary::Fixed;
        let uplift = self.params.uplift_m_per_year;
        let (mut ring, mut free) = (0.0, 0usize);
        for (i, (h, t)) in self.height.iter_mut().zip(&self.tendency).enumerate() {
            if fixed && self.grid.on_ring(i) {
                ring += t;
            } else {
                *h += dt * (t + uplift);
                free += 1;
            }
        }
        let area = self.grid.cell_area();
        self.exchange_m3 -= dt * ring * area;
        self.sources_m3 += dt * uplift * area * free as f64;
        self.time_years += dt;
    }

    fn report(&self, call: &Call) -> Diagnostics {
        let area = self.grid.cell_area();
        let integral = self.grid.integral(&self.height);
        let (mut east, mut south) = (Vec::new(), Vec::new());
        face_gradients(
            &self.grid,
            self.boundary.periodic(),
            &self.height,
            &mut east,
            &mut south,
        );
        let (lo, hi) = min_max(&self.height);
        let residual = integral - self.initial_integral - self.exchange_m3 - self.sources_m3;
        let scale = self.height.iter().map(|v| v.abs()).sum::<f64>() * area;
        Diagnostics {
            time_years: self.time_years,
            advanced_years: call.advanced,
            steps_done: call.steps_done,
            substeps: call.substeps,
            max_substep_years: call.max_substep,
            stable_dt_years: self.last_bound,
            integral_m3: integral,
            initial_integral_m3: self.initial_integral,
            boundary_exchange_m3: self.exchange_m3,
            sources_m3: self.sources_m3,
            residual_m3: residual,
            residual_relative: if scale > 0.0 { residual / scale } else { 0.0 },
            max_slope: max_abs(&east).max(max_abs(&south)),
            min_height_m: lo,
            max_height_m: hi,
            limited_faces: self.limited_faces,
            rejections: self.rejections,
            rejected: call.rejected,
            truncated: call.truncated,
            message: call.message.clone(),
        }
    }
}
