//! Linear diffusion and the critical-slope teacher, both in face form.

use crate::grid::{assemble, face_gradients, Face, Grid};

/// The teacher clips `|g| / S_c` to this value, as `hybrid.nonlinear_teacher` does.
pub const RATIO_CLIP: f64 = 0.99;

/// Parameters of the critical-slope teacher.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Teacher {
    /// D, m^2/yr.
    pub diffusivity: f64,
    /// S_c, rise over run.
    pub critical_slope: f64,
}

/// `2 a b / (a + b)`, or 0 when both are 0. The diffusivity of a face between
/// cells with diffusivities `a` and `b`.
pub fn harmonic_mean(a: f64, b: f64) -> f64 {
    let total = a + b;
    if total > 0.0 {
        2.0 * a * b / total
    } else {
        0.0
    }
}

/// Linear diffusion with uniform `d`: `out = d * div(grad h)`.
///
/// Same arithmetic as `hybrid.flux_divergence`.
pub fn linear_tendency(
    grid: &Grid,
    periodic: bool,
    height: &[f64],
    d: f64,
    east: &mut Vec<f64>,
    south: &mut Vec<f64>,
    out: &mut [f64],
) {
    face_gradients(grid, periodic, height, east, south);
    assemble(grid, periodic, east, south, out);
    out.iter_mut().for_each(|v| *v *= d);
}

/// Linear diffusion with a per-cell diffusivity `k`. Each face uses the
/// harmonic mean of its two cells, so the update still conserves exactly.
pub fn variable_tendency(
    grid: &Grid,
    periodic: bool,
    height: &[f64],
    k: &[f64],
    east: &mut Vec<f64>,
    south: &mut Vec<f64>,
    out: &mut [f64],
) {
    face_gradients(grid, periodic, height, east, south);
    grid.for_each_face(periodic, |face, a, b| {
        let d = harmonic_mean(k[a], k[b]);
        match face {
            Face::East(i) => east[i] *= d,
            Face::South(i) => south[i] *= d,
        }
    });
    assemble(grid, periodic, east, south, out);
}

/// The critical-slope teacher, `G = D g / (1 - r^2)` with
/// `r = clamp(|g| / S_c, 0, 0.99)`. Same arithmetic as
/// `hybrid.nonlinear_teacher`, including the clip. Returns how many faces
/// were clipped.
pub fn teacher_tendency(
    grid: &Grid,
    periodic: bool,
    height: &[f64],
    teacher: &Teacher,
    east: &mut Vec<f64>,
    south: &mut Vec<f64>,
    out: &mut [f64],
) -> usize {
    face_gradients(grid, periodic, height, east, south);
    let mut clipped = 0;
    for g in east.iter_mut().chain(south.iter_mut()) {
        let raw = g.abs() / teacher.critical_slope;
        clipped += usize::from(raw > RATIO_CLIP);
        let ratio = raw.clamp(0.0, RATIO_CLIP);
        *g /= 1.0 - ratio * ratio;
    }
    assemble(grid, periodic, east, south, out);
    out.iter_mut().for_each(|v| *v *= teacher.diffusivity);
    clipped
}

/// Explicit Euler bound for diffusion, `dx^2 / (4 D_max)`; infinite when `D_max` is 0.
///
/// With every face diffusivity at most `D_max`, the update writes each new
/// height as a convex combination of the old height and its neighbours once
/// `dt` is below this bound, so no new extrema appear.
pub fn linear_stable_dt(spacing_m: f64, d_max: f64) -> f64 {
    if d_max > 0.0 {
        spacing_m * spacing_m / (4.0 * d_max)
    } else {
        f64::INFINITY
    }
}

/// Explicit Euler bound for the teacher.
///
/// The teacher is diffusion with a face diffusivity `D / (1 - r^2)` that
/// grows with slope and is capped by the clip at `D / (1 - 0.99^2)`, about
/// `50 D`. Using that cap in [`linear_stable_dt`] gives the same convex
/// combination argument for any state.
pub fn teacher_stable_dt(spacing_m: f64, teacher: &Teacher) -> f64 {
    let cap = teacher.diffusivity / (1.0 - RATIO_CLIP * RATIO_CLIP);
    linear_stable_dt(spacing_m, cap)
}
