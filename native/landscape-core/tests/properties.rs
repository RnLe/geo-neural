//! Properties that hold without a reference run.

mod common;

use common::{closure, mass_scale};
use landscape_core::*;

fn grid() -> Grid {
    Grid::new(24, 50.0).unwrap()
}

/// A rough surface with slopes inside the closures' training range.
fn rough(grid: &Grid) -> Vec<f64> {
    let n = grid.side as f64;
    (0..grid.cells())
        .map(|i| {
            let (r, c) = ((i / grid.side) as f64, (i % grid.side) as f64);
            20.0 * (0.7 * c).sin() * (0.4 * r).cos() + 15.0 * (c / n) + 5.0 * ((r * c) * 0.37).sin()
        })
        .collect()
}

fn hill(grid: &Grid, sign: f64) -> Vec<f64> {
    let mid = (grid.side as f64 - 1.0) / 2.0;
    (0..grid.cells())
        .map(|i| {
            let (r, c) = ((i / grid.side) as f64 - mid, (i % grid.side) as f64 - mid);
            sign * 30.0 * (-(r * r + c * c) / 18.0).exp()
        })
        .collect()
}

fn run(model: Model, boundary: Boundary, initial: &[f64]) -> Scenario {
    let params = Params::defaults(&model);
    Scenario::new(grid(), boundary, model, params, initial).unwrap()
}

fn linear(d: f64) -> Model {
    Model::Linear {
        diffusivity: d,
        field: None,
    }
}

fn varying(grid: &Grid) -> Model {
    let field = (0..grid.cells())
        .map(|i| 0.01 + 0.2 * ((i as f64) * 0.618).fract())
        .collect();
    Model::Linear {
        diffusivity: 0.0,
        field: Some(field),
    }
}

fn teacher() -> Model {
    Model::Nonlinear(Teacher {
        diffusivity: 0.05,
        critical_slope: 0.6,
    })
}

/// Layers with deterministic pseudo-random weights in [-0.2, 0.2).
fn arbitrary_layers(shapes: &[(usize, usize)], last: Activation) -> (Vec<LayerSpec>, Vec<f32>) {
    let mut state = 12345u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.4
    };
    let mut weights = Vec::new();
    let mut specs = Vec::new();
    for (i, &(cin, cout)) in shapes.iter().enumerate() {
        let weight_offset = weights.len();
        weights.extend((0..cin * cout * 9).map(|_| next()));
        let bias_offset = weights.len();
        weights.extend((0..cout).map(|_| next()));
        let activation = if i + 1 == shapes.len() {
            last
        } else {
            Activation::Gelu
        };
        specs.push(LayerSpec {
            cin,
            cout,
            weight_offset,
            bias_offset,
            activation,
        });
    }
    (specs, weights)
}

/// A conductance closure with arbitrary weights: its guarantees must not depend on training.
fn arbitrary_conductance() -> Closure {
    let (specs, weights) = arbitrary_layers(&[(1, 8), (8, 8), (8, 1)], Activation::Sigmoid);
    let teacher = Teacher {
        diffusivity: 0.05,
        critical_slope: 0.6,
    };
    let open = Validated {
        max_slope: f64::INFINITY,
        min_height_m: f64::NEG_INFINITY,
        max_height_m: f64::INFINITY,
    };
    let apply = Apply::conductance(0.05, 3.0).unwrap();
    Closure::from_f32(apply, &specs, &weights, 50.0, teacher, open).unwrap()
}

/// A flux closure with arbitrary weights, to show conservation does not depend on training.
fn arbitrary_flux() -> Closure {
    let (specs, weights) = arbitrary_layers(&[(2, 8), (8, 8), (8, 1)], Activation::Identity);
    let teacher = Teacher {
        diffusivity: 0.05,
        critical_slope: 0.6,
    };
    let open = Validated {
        max_slope: f64::INFINITY,
        min_height_m: f64::NEG_INFINITY,
        max_height_m: f64::INFINITY,
    };
    Closure::from_f32(Apply::Flux, &specs, &weights, 50.0, teacher, open).unwrap()
}

#[test]
fn erf_and_activations_match_reference_values() {
    // From Python's math.erf. The series and continued fraction stay within a few ulp.
    let cases = [
        (0.1, 0.1124629160182849),
        (0.5, 0.5204998778130465),
        (1.0, 0.8427007929497149),
        (2.0, 0.9953222650189527),
        (3.0, 0.9999779095030014),
        (-1.5, -0.9661051464753108),
    ];
    for (x, expected) in cases {
        let error = (erf(x) - expected).abs() / expected.abs();
        println!("erf({x}): relative error {error:e}");
        assert!(error <= 2e-15, "erf({x})");
    }
    assert_eq!(erf(0.0), 0.0);
    assert_eq!(erf(7.0), 1.0);
    assert_eq!(gelu(0.0), 0.0);
    assert!((gelu(1.0) - 0.8413447460685429).abs() < 1e-15);
    assert_eq!(softplus(25.0), 25.0);
    assert_eq!(sigmoid(0.0), 0.5);
    assert!((sigmoid(2.0) - 0.8807970779778823).abs() < 1e-15);
    assert!((sigmoid(-800.0)).abs() < 1e-300 && sigmoid(800.0) == 1.0);
    assert!((softplus(0.0) - std::f64::consts::LN_2).abs() < 1e-16);
}

#[test]
fn a_constant_surface_does_not_move() {
    let g = grid();
    let flat = vec![12.5; g.cells()];
    let models = [
        (linear(0.05), Boundary::Closed),
        (linear(0.05), Boundary::Fixed),
        (linear(0.05), Boundary::Periodic),
        (varying(&g), Boundary::Closed),
        (teacher(), Boundary::Periodic),
        (Model::Learned(closure("kfield")), Boundary::Closed),
        (Model::Learned(closure("penalty")), Boundary::Fixed),
        (Model::Learned(closure("conductance")), Boundary::Closed),
        (Model::Learned(closure("conductance")), Boundary::Fixed),
        (Model::Learned(arbitrary_conductance()), Boundary::Closed),
    ];
    for (model, boundary) in models {
        let mut s = run(model, boundary, &flat);
        let d = s.step(3, 200.0);
        assert!(!d.rejected, "{}", d.message);
        assert_eq!(s.height(), &flat[..]);
        assert_eq!(d.boundary_exchange_m3, 0.0);
    }
}

#[test]
fn the_flux_arm_moves_a_flat_surface_only_at_closed_edges() {
    // On a flat surface every face sees the same input, so the network emits
    // one value c on every face. Interior cells cancel exactly; the edge cells
    // keep +-c/dx. That is the trained network's output at zero gradient, a
    // property of the weights, and it still conserves.
    let g = grid();
    let flat = vec![12.5; g.cells()];
    let mut tendency = vec![0.0; g.cells()];
    closure("flux").tendency(&g, &flat, &mut Work::default(), &mut tendency);
    for (i, t) in tendency.iter().enumerate() {
        if !g.on_ring(i) {
            assert_eq!(*t, 0.0);
        }
    }
    let edge = tendency.iter().fold(0.0f64, |m, t| m.max(t.abs()));
    println!("flux arm on a flat surface: largest edge tendency {edge:e} m/yr");
    assert!(tendency.iter().sum::<f64>().abs() <= 1e-15 * edge * g.cells() as f64);
}

#[test]
fn closed_domains_conserve_to_rounding() {
    let g = grid();
    let start = rough(&g);
    let models = [
        ("linear", linear(0.05)),
        ("variable D", varying(&g)),
        ("teacher", teacher()),
        ("flux arm", Model::Learned(closure("flux"))),
        (
            "flux arm, arbitrary weights",
            Model::Learned(arbitrary_flux()),
        ),
        ("conductance arm", Model::Learned(closure("conductance"))),
        (
            "conductance arm, arbitrary weights",
            Model::Learned(arbitrary_conductance()),
        ),
    ];
    for (name, model) in models {
        let mut s = run(model, Boundary::Closed, &start);
        let d = s.step(6, 200.0);
        assert!(!d.rejected, "{name}: {}", d.message);
        println!(
            "{name}: residual {:e} of sum|h|a after {} substeps",
            d.residual_relative, d.substeps
        );
        assert!(d.residual_relative.abs() <= 1e-14, "{name}");
        assert_ne!(s.height(), &start[..], "{name} should change the surface");
    }
    // The K-field arm is declared non-conservative and the ledger shows it.
    let mut s = run(Model::Learned(closure("kfield")), Boundary::Closed, &start);
    let d = s.step(6, 100.0);
    println!("kfield arm: residual {:e} of sum|h|a", d.residual_relative);
    assert!(d.residual_relative.abs() > 1e-10);
}

#[test]
fn the_conductance_arm_is_offset_invariant_dissipative_and_bounded_for_any_weights() {
    let g = grid();
    let start = rough(&g);
    for closure in [arbitrary_conductance(), closure("conductance")] {
        // Adding a constant leaves every face gradient, and so the tendency, unchanged.
        let shifted: Vec<f64> = start.iter().map(|h| h + 100.0).collect();
        let (mut a, mut b) = (vec![0.0; g.cells()], vec![0.0; g.cells()]);
        let bound = closure.tendency(&g, &start, &mut Work::default(), &mut a);
        closure.tendency(&g, &shifted, &mut Work::default(), &mut b);
        let scale = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let offset = a
            .iter()
            .zip(&b)
            .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
        assert!(
            offset <= 1e-12 * scale,
            "offset changed the tendency by {offset:e}"
        );
        assert_eq!(bound, Some(3.0));
        // Below the bound the update is a convex combination: the energy about the
        // mean never increases and no new extrema appear.
        let mut s = run(Model::Learned(closure), Boundary::Closed, &start);
        let mean = start.iter().sum::<f64>() / start.len() as f64;
        let energy = |h: &[f64]| h.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>();
        let mut previous = energy(&start);
        let (lo, hi) = start
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| {
                (l.min(v), h.max(v))
            });
        for _ in 0..10 {
            let d = s.step(1, 500.0);
            assert!(!d.rejected, "{}", d.message);
            assert!(d.max_substep_years <= 0.9 * 2500.0 / 12.0 * (1.0 + 1e-12));
            let now = energy(s.height());
            assert!(
                now <= previous * (1.0 + 1e-14),
                "energy rose from {previous} to {now}"
            );
            assert!(d.min_height_m >= lo - 1e-9 && d.max_height_m <= hi + 1e-9);
            previous = now;
        }
    }
}

#[test]
fn fixed_edges_book_what_leaves_and_hold_the_ring() {
    let g = grid();
    for (sign, model) in [
        (1.0, linear(0.05)),
        (-1.0, linear(0.05)),
        (1.0, teacher()),
        (1.0, Model::Learned(closure("flux"))),
    ] {
        let start = hill(&g, sign);
        let mut s = run(model, Boundary::Fixed, &start);
        let d = s.step(20, 200.0);
        assert!(!d.rejected, "{}", d.message);
        // A hill sheds material through the base level, a pit draws it in.
        assert!(sign * d.boundary_exchange_m3 < -1e-6 * mass_scale(&start, g.spacing_m));
        assert!(d.residual_relative.abs() <= 1e-14);
        for i in (0..g.cells()).filter(|&i| g.on_ring(i)) {
            assert_eq!(s.height()[i], start[i]);
        }
    }
    let mut closed = run(linear(0.05), Boundary::Closed, &hill(&g, 1.0));
    assert_eq!(closed.step(20, 200.0).boundary_exchange_m3, 0.0);
}

#[test]
fn uplift_is_booked_as_a_source() {
    let g = grid();
    let params = Params {
        uplift_m_per_year: 1e-3,
        ..Params::defaults(&teacher())
    };
    let mut s = Scenario::new(g, Boundary::Fixed, teacher(), params, &rough(&g)).unwrap();
    let d = s.step(10, 100.0);
    let free = (0..g.cells()).filter(|&i| !g.on_ring(i)).count() as f64;
    assert!((d.sources_m3 - 1e-3 * 1000.0 * g.cell_area() * free).abs() <= 1e-9 * d.sources_m3);
    assert!(d.residual_relative.abs() <= 1e-14);
}

#[test]
fn substeps_respect_the_stability_bound() {
    let g = grid();
    let checker: Vec<f64> = (0..g.cells())
        .map(|i| {
            if (i / g.side + i % g.side).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            }
        })
        .collect();
    let bound = linear_stable_dt(g.spacing_m, 0.05);
    let dt = 8.0 * bound;
    let mut s = run(linear(0.05), Boundary::Periodic, &checker);
    let d = s.step(1, dt);
    assert_eq!(d.stable_dt_years, bound);
    assert_eq!(d.substeps, (8.0f64 / 0.9).ceil() as u32);
    assert!(d.max_substep_years <= 0.9 * bound);
    assert!((d.advanced_years - dt).abs() <= 1e-12 * dt);
    // Below the bound the update is a convex combination: no new extrema.
    assert!(d.max_height_m <= 1.0 && d.min_height_m >= -1.0);

    // One unsplit step at the same dt amplifies the checkerboard 15-fold.
    let mut tendency = vec![0.0; g.cells()];
    linear_tendency(
        &g,
        true,
        &checker,
        0.05,
        &mut Vec::new(),
        &mut Vec::new(),
        &mut tendency,
    );
    let unsplit = checker
        .iter()
        .zip(&tendency)
        .map(|(h, t)| (h + dt * t).abs())
        .fold(0.0, f64::max);
    assert!((unsplit - 15.0).abs() < 1e-9);

    // The teacher's bound comes from the diffusivity at the slope clip.
    let start = rough(&g);
    let mut s = run(teacher(), Boundary::Closed, &start);
    let d = s.step(1, 1000.0);
    let bound = teacher_stable_dt(
        g.spacing_m,
        &Teacher {
            diffusivity: 0.05,
            critical_slope: 0.6,
        },
    );
    assert!((bound - 2500.0 * (1.0 - 0.99f64 * 0.99) / 0.2).abs() < 1e-9);
    assert_eq!(d.substeps, (1000.0 / (0.9 * bound)).ceil() as u32);
    assert!(d.max_substep_years <= 0.9 * bound);
    let (lo, hi) = start
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| {
            (l.min(v), h.max(v))
        });
    assert!(d.min_height_m >= lo && d.max_height_m <= hi);

    // The K-field bound follows the predicted diffusivity.
    let mut s = run(Model::Learned(closure("kfield")), Boundary::Closed, &start);
    let d = s.step(1, 2000.0);
    assert!(!d.rejected, "{}", d.message);
    assert!(d.max_substep_years <= 0.9 * d.stable_dt_years * (1.0 + 1e-12));
    assert!(d.substeps > 1);
}

#[test]
fn a_call_stops_at_the_substep_cap() {
    let g = grid();
    let model = linear(0.05);
    let params = Params {
        max_substeps: 4,
        ..Params::defaults(&model)
    };
    let mut s = Scenario::new(g, Boundary::Closed, model, params, &rough(&g)).unwrap();
    let bound = linear_stable_dt(g.spacing_m, 0.05);
    let d = s.step(3, 20.0 * bound);
    assert!(d.truncated && !d.rejected);
    assert_eq!((d.substeps, d.steps_done), (4, 0));
    assert!((d.advanced_years - 4.0 * 20.0 * bound / 23.0).abs() < 1e-9 * bound);
    assert_eq!(d.time_years, d.advanced_years);
    assert_eq!(
        Params::defaults(&Model::Learned(closure("flux"))).max_substeps,
        4
    );
}

#[test]
fn learned_closures_refuse_states_outside_their_training_range() {
    let g = grid();
    let steep: Vec<f64> = (0..g.cells())
        .map(|i| 150.0 * ((i % g.side) % 2) as f64)
        .collect();
    let high = vec![500.0; g.cells()];
    for (start, what) in [(steep, "slope"), (high, "heights")] {
        let mut s = run(Model::Learned(closure("flux")), Boundary::Closed, &start);
        let d = s.step(2, 200.0);
        assert!(d.rejected, "{what} should be refused");
        assert!(d.message.contains(what), "{}", d.message);
        assert_eq!((d.substeps, d.rejections), (0, 1));
        assert_eq!(s.height(), &start[..]);
    }
    // The teacher clips as hybrid.nonlinear_teacher does, and counts it.
    let steep: Vec<f64> = (0..g.cells())
        .map(|i| 150.0 * ((i % g.side) % 2) as f64)
        .collect();
    let mut s = run(teacher(), Boundary::Closed, &steep);
    let d = s.step(1, 10.0);
    assert!(!d.rejected && d.limited_faces > 0);
}

#[test]
fn bad_inputs_are_refused() {
    let g = grid();
    let start = rough(&g);
    let params = Params::defaults(&linear(0.05));
    let make =
        |boundary, model, initial: &[f64]| Scenario::new(g, boundary, model, params, initial);
    assert!(make(Boundary::Periodic, Model::Learned(closure("flux")), &start).is_err());
    let coarse = Grid::new(24, 100.0).unwrap();
    assert!(Scenario::new(
        coarse,
        Boundary::Closed,
        Model::Learned(closure("flux")),
        params,
        &start
    )
    .is_err());
    assert!(make(Boundary::Closed, linear(0.05), &start[1..]).is_err());
    let mut broken = start.clone();
    broken[3] = f64::NAN;
    assert!(make(Boundary::Closed, linear(0.05), &broken).is_err());
    assert!(make(Boundary::Closed, linear(-1.0), &start).is_err());
    let negative = Model::Linear {
        diffusivity: 0.0,
        field: Some(vec![-1.0; g.cells()]),
    };
    assert!(make(Boundary::Closed, negative, &start).is_err());
    assert!(Boundary::parse("open").is_err());
    let mut s = make(Boundary::Closed, linear(0.05), &start).unwrap();
    let d = s.step(1, f64::NAN);
    assert!(d.rejected && d.rejections == 1);
    assert!(s.reset(&start[..10]).is_err());
    s.reset(&start).unwrap();
    assert_eq!(s.diagnostics().rejections, 0);
}
