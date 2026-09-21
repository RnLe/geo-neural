//! Grid geometry, boundary modes and the shared face assembly.

use crate::{fail, Error};

/// How the domain edge behaves. See the crate docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boundary {
    Closed,
    Fixed,
    Periodic,
}

impl Boundary {
    /// Parses `"closed"`, `"fixed"` or `"periodic"`.
    pub fn parse(name: &str) -> Result<Self, Error> {
        match name {
            "closed" => Ok(Self::Closed),
            "fixed" => Ok(Self::Fixed),
            "periodic" => Ok(Self::Periodic),
            _ => fail(format!(
                "unknown boundary {name:?}; use closed, fixed or periodic"
            )),
        }
    }

    pub fn periodic(self) -> bool {
        self == Self::Periodic
    }
}

/// A square grid of `side * side` cells of edge `spacing_m`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid {
    pub side: usize,
    pub spacing_m: f64,
}

impl Grid {
    pub fn new(side: usize, spacing_m: f64) -> Result<Self, Error> {
        if side < 2 {
            return fail("side must be at least 2");
        }
        if !(spacing_m.is_finite() && spacing_m > 0.0) {
            return fail("spacing_m must be positive and finite");
        }
        Ok(Self { side, spacing_m })
    }

    pub fn cells(&self) -> usize {
        self.side * self.side
    }

    pub fn cell_area(&self) -> f64 {
        self.spacing_m * self.spacing_m
    }

    /// `sum h_i * a`, in cubic metres.
    pub fn integral(&self, height: &[f64]) -> f64 {
        height.iter().sum::<f64>() * self.cell_area()
    }

    /// True for cells in the outermost row or column.
    pub fn on_ring(&self, index: usize) -> bool {
        let (r, c) = (index / self.side, index % self.side);
        r == 0 || c == 0 || r + 1 == self.side || c + 1 == self.side
    }

    /// Columns of east faces: `side - 1`, or `side` when periodic.
    pub fn east_cols(&self, periodic: bool) -> usize {
        if periodic {
            self.side
        } else {
            self.side - 1
        }
    }

    /// Rows of south faces: `side - 1`, or `side` when periodic.
    pub fn south_rows(&self, periodic: bool) -> usize {
        self.east_cols(periodic)
    }

    /// Calls `visit(face, a, b)` for every east face, then every south face.
    /// `a` is the west or north cell and `b` its east or south neighbour; `face`
    /// indexes the east array, then the south array.
    pub(crate) fn for_each_face(&self, periodic: bool, mut visit: impl FnMut(Face, usize, usize)) {
        let n = self.side;
        let ec = self.east_cols(periodic);
        for r in 0..n {
            for c in 0..ec {
                visit(Face::East(r * ec + c), r * n + c, r * n + (c + 1) % n);
            }
        }
        for r in 0..self.south_rows(periodic) {
            for c in 0..n {
                visit(Face::South(r * n + c), r * n + c, ((r + 1) % n) * n + c);
            }
        }
    }
}

/// Index of a face in the east or south face array.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Face {
    East(usize),
    South(usize),
}

/// Face gradients `(h_b - h_a) / dx`, east faces (`side` rows by
/// [`Grid::east_cols`]) and south faces ([`Grid::south_rows`] by `side`).
pub fn face_gradients(
    grid: &Grid,
    periodic: bool,
    height: &[f64],
    east: &mut Vec<f64>,
    south: &mut Vec<f64>,
) {
    east.resize(grid.side * grid.east_cols(periodic), 0.0);
    south.resize(grid.south_rows(periodic) * grid.side, 0.0);
    let dx = grid.spacing_m;
    grid.for_each_face(periodic, |face, a, b| {
        let g = (height[b] - height[a]) / dx;
        match face {
            Face::East(i) => east[i] = g,
            Face::South(i) => south[i] = g,
        }
    });
}

/// Net inflow per cell from face values: `out[a] += G / dx`, `out[b] -= G / dx`.
///
/// The four passes run in the order of `hybrid.divergence`, so for the same
/// face values the result agrees with it to the bit.
pub fn assemble(grid: &Grid, periodic: bool, east: &[f64], south: &[f64], out: &mut [f64]) {
    let n = grid.side;
    let dx = grid.spacing_m;
    let ec = grid.east_cols(periodic);
    let sr = grid.south_rows(periodic);
    out.fill(0.0);
    for r in 0..n {
        for c in 0..ec {
            out[r * n + c] += east[r * ec + c] / dx;
        }
    }
    for r in 0..n {
        for c in 0..ec {
            out[r * n + (c + 1) % n] -= east[r * ec + c] / dx;
        }
    }
    for r in 0..sr {
        for c in 0..n {
            out[r * n + c] += south[r * n + c] / dx;
        }
    }
    for r in 0..sr {
        for c in 0..n {
            out[((r + 1) % n) * n + c] -= south[r * n + c] / dx;
        }
    }
}

/// Largest absolute value in a face array.
pub(crate) fn max_abs(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |m, v| m.max(v.abs()))
}
