//! Inference for the learned closures of `hybrid._modules`: stacks of 3x3
//! convolutions (stride 1, replicate padding), evaluated in `f64` from
//! `float32` weights.

use crate::grid::{assemble, max_abs, Grid};
use crate::physics::Teacher;
use crate::{fail, Error};

/// Error function, within a few ulp of a correctly rounded result.
///
/// Below 2.5 it sums `erf(x) = 2/sqrt(pi) exp(-x^2) sum (2x^2)^n x / (2n+1)!!`,
/// whose terms are all positive. From 2.5 to 6 it uses the continued fraction
/// for `erfc`. Beyond 6, `erf` rounds to 1.
pub fn erf(x: f64) -> f64 {
    let a = x.abs();
    let value = if a < 2.5 {
        let a2 = a * a;
        let (mut term, mut sum, mut n) = (a, a, 0.0);
        loop {
            n += 1.0;
            term *= 2.0 * a2 / (2.0 * n + 1.0);
            sum += term;
            if term <= sum * 1e-17 {
                break;
            }
        }
        std::f64::consts::FRAC_2_SQRT_PI * (-a2).exp() * sum
    } else if a < 6.0 {
        let mut f = a;
        for n in (1..=40).rev() {
            f = a + 0.5 * f64::from(n) / f;
        }
        1.0 - (-a * a).exp() / (std::f64::consts::PI.sqrt() * f)
    } else {
        1.0
    };
    value.copysign(x)
}

/// `torch.nn.GELU()`: `0.5 x (1 + erf(x / sqrt 2))`.
pub fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + erf(x * std::f64::consts::FRAC_1_SQRT_2))
}

/// `torch.nn.functional.softplus` with `beta = 1`, `threshold = 20`.
pub fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// Activation applied after a convolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Identity,
    Gelu,
    Softplus,
}

impl Activation {
    /// Parses `"identity"`, `"gelu"` or `"softplus"`.
    pub fn parse(name: &str) -> Result<Self, Error> {
        match name {
            "identity" => Ok(Self::Identity),
            "gelu" => Ok(Self::Gelu),
            "softplus" => Ok(Self::Softplus),
            _ => fail(format!("unknown activation {name:?}")),
        }
    }

    fn apply(self, x: f64) -> f64 {
        match self {
            Self::Identity => x,
            Self::Gelu => gelu(x),
            Self::Softplus => softplus(x),
        }
    }
}

/// How the network output becomes a tendency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Apply {
    /// Inputs (face gradient, face mean height) on east faces; output is the
    /// face value `G`. South faces reuse the network on the transposed field.
    /// Conservative for any weights.
    Flux,
    /// Input is cell height; output is a cell diffusivity `K` (the stack ends
    /// in softplus) used as `K * laplacian(h)` with replicate edges. Not
    /// conservative: a varying `K` outside the divergence does not telescope.
    KField,
}

impl Apply {
    /// Parses `"flux"` or `"kfield"`.
    pub fn parse(name: &str) -> Result<Self, Error> {
        match name {
            "flux" => Ok(Self::Flux),
            "kfield" => Ok(Self::KField),
            _ => fail(format!("unknown closure application {name:?}")),
        }
    }

    fn inputs(self) -> usize {
        match self {
            Self::Flux => 2,
            Self::KField => 1,
        }
    }
}

/// One 3x3 convolution with weights `[out][in][3][3]` and bias `[out]`.
#[derive(Clone, Debug)]
pub struct Conv {
    pub cin: usize,
    pub cout: usize,
    pub weight: Vec<f64>,
    pub bias: Vec<f64>,
    pub activation: Activation,
}

/// Where one layer sits in a flat `float32` weight array, in elements.
#[derive(Clone, Copy, Debug)]
pub struct LayerSpec {
    pub cin: usize,
    pub cout: usize,
    pub weight_offset: usize,
    pub bias_offset: usize,
    pub activation: Activation,
}

/// Range of the training data. A learned closure is only checked inside it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Validated {
    pub max_slope: f64,
    pub min_height_m: f64,
    pub max_height_m: f64,
}

/// A trained closure and the conditions it was trained under.
#[derive(Clone, Debug)]
pub struct Closure {
    pub apply: Apply,
    pub layers: Vec<Conv>,
    /// The spacing the network was trained at; it sees `dh/dx` at this spacing.
    pub spacing_m: f64,
    /// The teacher it imitates, used for the nominal stability bound.
    pub teacher: Teacher,
    pub validated: Validated,
}

impl Closure {
    /// Builds a closure from a flat weight array and its layer layout.
    pub fn from_f32(
        apply: Apply,
        specs: &[LayerSpec],
        weights: &[f32],
        spacing_m: f64,
        teacher: Teacher,
        validated: Validated,
    ) -> Result<Self, Error> {
        if specs.is_empty() {
            return fail("a closure needs at least one layer");
        }
        let mut channels = apply.inputs();
        let mut layers = Vec::with_capacity(specs.len());
        for (index, spec) in specs.iter().enumerate() {
            if spec.cin != channels || spec.cout == 0 {
                return fail(format!(
                    "layer {index} expects {} inputs, gets {channels}",
                    spec.cin
                ));
            }
            let take = |offset: usize, count: usize| -> Result<Vec<f64>, Error> {
                match weights.get(offset..offset + count) {
                    Some(s) => Ok(s.iter().map(|&w| f64::from(w)).collect()),
                    None => fail(format!("layer {index} lies outside the weight array")),
                }
            };
            layers.push(Conv {
                cin: spec.cin,
                cout: spec.cout,
                weight: take(spec.weight_offset, spec.cout * spec.cin * 9)?,
                bias: take(spec.bias_offset, spec.cout)?,
                activation: spec.activation,
            });
            channels = spec.cout;
        }
        if channels != 1 {
            return fail("the last layer must have one output channel");
        }
        let finite = layers
            .iter()
            .all(|l| l.weight.iter().chain(&l.bias).all(|w| w.is_finite()));
        if !finite {
            return fail("weights contain non-finite values");
        }
        Ok(Self {
            apply,
            layers,
            spacing_m,
            teacher,
            validated,
        })
    }

    /// Runs the stack on `input` (`cin` planes of `rows * cols`) and returns
    /// the single output plane, borrowed from `work`.
    pub fn forward<'w>(
        &self,
        input: &[f64],
        rows: usize,
        cols: usize,
        work: &'w mut Work,
    ) -> &'w [f64] {
        let Work { a, b, pad } = work;
        a.clear();
        a.extend_from_slice(input);
        for layer in &self.layers {
            conv(layer, a, rows, cols, pad, b);
            std::mem::swap(a, b);
        }
        &a[..rows * cols]
    }

    /// Fills `out` with the closure tendency on a closed domain (the boundary
    /// the closures were trained with). Returns the largest `K` for
    /// [`Apply::KField`], or `None` for [`Apply::Flux`].
    pub fn tendency(
        &self,
        grid: &Grid,
        height: &[f64],
        work: &mut Work,
        out: &mut [f64],
    ) -> Option<f64> {
        match self.apply {
            Apply::Flux => {
                self.flux(grid, height, work, out);
                None
            }
            Apply::KField => Some(self.kfield(grid, height, work, out)),
        }
    }

    /// `hybrid._apply_flux`: face fluxes both ways through one network.
    fn flux(&self, grid: &Grid, height: &[f64], work: &mut Work, out: &mut [f64]) {
        let n = grid.side;
        let ec = n - 1;
        let dx = grid.spacing_m;
        // Network inputs on the east faces of `at(r, c)`: gradient, then mean height.
        let faces = |at: &dyn Fn(usize, usize) -> f64, work: &mut Work| -> Vec<f64> {
            let mut input = vec![0.0; 2 * n * ec];
            for r in 0..n {
                for c in 0..ec {
                    let (left, right) = (at(r, c), at(r, c + 1));
                    input[r * ec + c] = (right - left) / dx;
                    input[n * ec + r * ec + c] = 0.5 * (right + left);
                }
            }
            self.forward(&input, n, ec, work).to_vec()
        };
        let east = faces(&|r, c| height[r * n + c], work);
        let turned = faces(&|r, c| height[c * n + r], work);
        // The transposed east faces are the south faces: south[r][c] = turned[c][r].
        let mut south = vec![0.0; ec * n];
        for r in 0..ec {
            for c in 0..n {
                south[r * n + c] = turned[c * ec + r];
            }
        }
        assemble(grid, false, &east, &south, out);
    }

    /// `hybrid._apply_kfield`: learned `K` times the five-point Laplacian with
    /// replicate edges. Returns the largest `K`.
    fn kfield(&self, grid: &Grid, height: &[f64], work: &mut Work, out: &mut [f64]) -> f64 {
        let n = grid.side;
        let k = self.forward(height, n, n, work);
        let dx2 = grid.spacing_m * grid.spacing_m;
        let at = |r: usize, c: usize| height[r.min(n - 1) * n + c.min(n - 1)];
        for r in 0..n {
            for c in 0..n {
                let up = at(r.saturating_sub(1), c);
                let down = at(r + 1, c);
                let left = at(r, c.saturating_sub(1));
                let right = at(r, c + 1);
                let curvature = (up + down + left + right - 4.0 * height[r * n + c]) / dx2;
                out[r * n + c] = k[r * n + c] * curvature;
            }
        }
        max_abs(k)
    }

    /// Checks a state against the training range; `Err` names the violation.
    pub fn check(&self, height: &[f64], max_slope: f64) -> Result<(), String> {
        let v = &self.validated;
        let (lo, hi) = min_max(height);
        if max_slope > v.max_slope {
            return Err(format!(
                "max slope {max_slope:.4} exceeds the validated {:.4} of the learned closure",
                v.max_slope
            ));
        }
        if lo < v.min_height_m || hi > v.max_height_m {
            return Err(format!(
                "heights [{lo:.2}, {hi:.2}] m leave the validated [{:.2}, {:.2}] m of the learned closure",
                v.min_height_m, v.max_height_m
            ));
        }
        Ok(())
    }
}

pub(crate) fn min_max(values: &[f64]) -> (f64, f64) {
    values
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        })
}

/// Scratch buffers reused across network evaluations.
#[derive(Clone, Debug, Default)]
pub struct Work {
    a: Vec<f64>,
    b: Vec<f64>,
    pad: Vec<f64>,
}

/// One convolution with replicate padding, then the activation.
fn conv(
    layer: &Conv,
    input: &[f64],
    rows: usize,
    cols: usize,
    pad: &mut Vec<f64>,
    out: &mut Vec<f64>,
) {
    let (pr, pc) = (rows + 2, cols + 2);
    let plane = rows * cols;
    pad.resize(layer.cin * pr * pc, 0.0);
    for i in 0..layer.cin {
        for r in 0..pr {
            let sr = r.saturating_sub(1).min(rows - 1);
            for c in 0..pc {
                let sc = c.saturating_sub(1).min(cols - 1);
                pad[(i * pr + r) * pc + c] = input[i * plane + sr * cols + sc];
            }
        }
    }
    out.resize(layer.cout * plane, 0.0);
    for o in 0..layer.cout {
        let dst = &mut out[o * plane..(o + 1) * plane];
        dst.fill(layer.bias[o]);
        for i in 0..layer.cin {
            let src = &pad[i * pr * pc..(i + 1) * pr * pc];
            for ky in 0..3 {
                for kx in 0..3 {
                    let w = layer.weight[((o * layer.cin + i) * 3 + ky) * 3 + kx];
                    for r in 0..rows {
                        let s = &src[(r + ky) * pc + kx..][..cols];
                        for (d, s) in dst[r * cols..][..cols].iter_mut().zip(s) {
                            *d += w * s;
                        }
                    }
                }
            }
        }
        dst.iter_mut().for_each(|v| *v = layer.activation.apply(*v));
    }
}
