//! Hillslope evolution on a square grid: linear diffusion, the critical-slope
//! teacher from `geoneural.physics.hybrid`, and inference for the closures
//! trained against it.
//!
//! # Grid semantics
//!
//! A grid holds `side * side` heights in row-major order, `h[r * side + c]`,
//! with row 0 on the north edge. Each value is the mean height of a square cell
//! of edge `spacing_m` centred on a lattice node, so the node and cell readings
//! are the same numbers. Every cell, the outermost ring included, owns the full
//! area `a = spacing_m^2`: the domain edge lies half a spacing outside the
//! outermost nodes, and the integrated height is `sum h_i * a`.
//!
//! Transport lives on faces. Each pair of edge-adjacent cells shares one face
//! carrying one value `G` (m^2/yr). Per unit time the west or north cell gains
//! `G / dx` and its east or south neighbour loses the same amount. Any face
//! field therefore moves material without creating it, which is why linear
//! diffusion with a variable diffusivity and the flux closure conserve to
//! rounding.
//!
//! Boundary modes ([`Boundary`]):
//! - `Closed`: no faces on the domain edge, so zero flux. This is the same
//!   operator as a Laplacian with edge-replicated padding.
//! - `Fixed`: the outermost ring keeps its initial height, a fixed base level.
//!   Whatever the interior moves into the ring leaves the domain and is
//!   counted as boundary exchange.
//! - `Periodic`: the last column faces the first and the last row the first.
//!
//! Units: metres, years, m^2/yr. State is `f64` throughout.

mod closure;
mod grid;
mod physics;
mod scenario;

pub use closure::{
    erf, gelu, softplus, Activation, Apply, Closure, Conv, LayerSpec, Validated, Work,
};
pub use grid::{assemble, face_gradients, Boundary, Grid};
pub use physics::{
    harmonic_mean, linear_stable_dt, linear_tendency, teacher_stable_dt, teacher_tendency,
    variable_tendency, Teacher, RATIO_CLIP,
};
pub use scenario::{Diagnostics, Model, Params, Scenario};

/// A rejected input, with a message meant for a person.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub(crate) fn fail<T>(message: impl Into<String>) -> Result<T, Error> {
    Err(Error(message.into()))
}
