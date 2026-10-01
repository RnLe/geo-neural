//! Rust against the Python fixtures in `native/fixtures`.
//!
//! Linear and teacher cases run the same f64 arithmetic in the same order as
//! numpy, so they are held to 1e-12. Learned arms are held to 1e-9 against the
//! torch float64 rollout and to 1e-5 of the largest height against the torch
//! float32 rollout (see `learned_arm`).

mod common;

use common::*;
use landscape_core::{Closure, Grid, Model, Teacher, Work};

/// Steps one at a time and returns the integral after each step.
fn integrals(scenario: &mut landscape_core::Scenario, steps: usize, dt: f64) -> Vec<f64> {
    (0..steps)
        .map(|_| {
            let d = scenario.step(1, dt);
            assert!(!d.rejected && !d.truncated, "{}", d.message);
            assert_eq!(d.substeps, 1, "a fixture step must not be split");
            d.integral_m3
        })
        .collect()
}

/// Largest `|a - b|` over the scale `sum |h| a`.
fn ledger_error(a: &[f64], b: &[f64], scale: f64) -> f64 {
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
        / scale
}

#[test]
fn sine_mode_matches_numpy_and_decays_at_the_analytic_rate() {
    let case = load("linear-sine");
    let (d, dt, dx) = (
        num(&case, "diffusivity"),
        num(&case, "dtYears"),
        num(&case, "spacingM"),
    );
    let steps = case["steps"].as_u64().unwrap() as u32;
    let mut s = scenario(
        &case,
        Model::Linear {
            diffusivity: d,
            field: None,
        },
    );
    let report = s.step(steps, dt);
    assert_eq!(report.substeps, steps);
    let error = relative(s.height(), &f64s(&case["final"]));
    println!("sine: max error vs numpy {error:e} of max height");
    assert!(error <= 1e-12);

    // Amplitude by projection on the initial mode.
    let start = f64s(&case["initial"]);
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let ratio = dot(s.height(), &start) / dot(&start, &start);

    let side = case["side"].as_u64().unwrap() as f64;
    let length = side * dx;
    let kx = 2.0 * std::f64::consts::PI * num(&case, "modeX") / length;
    let ky = 2.0 * std::f64::consts::PI * num(&case, "modeY") / length;
    let k2 = kx * kx + ky * ky;
    // The five-point eigenvalue and the exact Euler amplification of this mode.
    let eigen = 4.0 / (dx * dx) * ((kx * dx / 2.0).sin().powi(2) + (ky * dx / 2.0).sin().powi(2));
    let discrete = (1.0 - dt * d * eigen).powi(steps as i32);
    println!("sine: amplitude ratio {ratio}, discrete prediction {discrete}");
    assert!((ratio / discrete - 1.0).abs() <= 1e-10);

    // Against exp(-D k^2 t): the rate differs by the leading spatial error
    // -(kx^4 + ky^4) dx^2 / (12 k^2) plus the Euler error dt D k^2 / 2.
    let t = dt * f64::from(steps);
    let rate_error = (-ratio.ln() / t) / (d * k2) - 1.0;
    let predicted = -(kx.powi(4) + ky.powi(4)) * dx * dx / (12.0 * k2) + dt * d * k2 / 2.0;
    println!(
        "sine: rate error vs exp(-D k^2 t) {rate_error:e}, leading-order prediction {predicted:e}"
    );
    assert!(rate_error.abs() < 5e-3);
    assert!((rate_error - predicted).abs() < 0.05 * predicted.abs());
}

#[test]
fn closed_linear_matches_flux_divergence_and_conserves() {
    let case = load("linear-closed");
    let (d, dt, dx) = (
        num(&case, "diffusivity"),
        num(&case, "dtYears"),
        num(&case, "spacingM"),
    );
    let steps = case["steps"].as_u64().unwrap() as usize;
    let mut s = scenario(
        &case,
        Model::Linear {
            diffusivity: d,
            field: None,
        },
    );
    let ours = integrals(&mut s, steps, dt);
    let error = relative(s.height(), &f64s(&case["final"]));
    let scale = mass_scale(s.height(), dx);
    let ledger = ledger_error(&ours, &f64s(&case["integrals"]), scale);
    let report = s.diagnostics();
    println!(
        "closed linear: field error {error:e}, integral error {ledger:e}, residual {:e}",
        report.residual_relative
    );
    assert!(error <= 1e-12);
    assert!(ledger <= 1e-14);
    assert!(report.residual_relative.abs() <= 1e-14);
}

#[test]
fn variable_diffusivity_matches_numpy_and_conserves() {
    let case = load("linear-variable");
    let (dt, dx) = (num(&case, "dtYears"), num(&case, "spacingM"));
    let steps = case["steps"].as_u64().unwrap() as usize;
    let field = f64s(&case["field"]);
    let mut s = scenario(
        &case,
        Model::Linear {
            diffusivity: 0.0,
            field: Some(field),
        },
    );
    let ours = integrals(&mut s, steps, dt);
    let error = relative(s.height(), &f64s(&case["final"]));
    let ledger = ledger_error(&ours, &f64s(&case["integrals"]), mass_scale(s.height(), dx));
    let report = s.diagnostics();
    println!(
        "variable D: field error {error:e}, integral error {ledger:e}, residual {:e}",
        report.residual_relative
    );
    assert!(error <= 1e-12);
    assert!(ledger <= 1e-14);
    assert!(report.residual_relative.abs() <= 1e-14);
}

#[test]
fn fixed_edges_match_landscape_evolve() {
    let case = load("linear-fixed");
    let (d, dt, dx) = (
        num(&case, "diffusivity"),
        num(&case, "dtYears"),
        num(&case, "spacingM"),
    );
    let steps = case["steps"].as_u64().unwrap() as u32;
    let mut s = scenario(
        &case,
        Model::Linear {
            diffusivity: d,
            field: None,
        },
    );
    let report = s.step(steps, dt);
    let error = relative(s.height(), &f64s(&case["final"]));
    let outflow = num(&case, "boundaryOutflowVolumeM3");
    let exchange_error = (report.boundary_exchange_m3 + outflow).abs() / outflow.abs();
    println!(
        "fixed edges: field error {error:e}, exchange {:.6e} m3 vs outflow {outflow:.6e} m3 (error {exchange_error:e}), residual {:e}",
        report.boundary_exchange_m3, report.residual_relative
    );
    assert_eq!(report.substeps, steps);
    // landscape.laplacian sums neighbours directly rather than through faces.
    assert!(error <= 1e-12);
    // The hill must lose a visible share of the material, not rounding.
    assert!(
        outflow > 1e-3 * mass_scale(s.height(), dx),
        "material should leave through the base level"
    );
    assert!(exchange_error <= 1e-9);
    assert!(report.residual_relative.abs() <= 1e-14);
}

#[test]
fn teacher_matches_python_and_reports_the_limiter() {
    let case = load("teacher");
    let teacher = Teacher {
        diffusivity: num(&case, "diffusivity"),
        critical_slope: num(&case, "criticalSlope"),
    };
    let (dt, dx) = (num(&case, "dtYears"), num(&case, "spacingM"));
    let steps = case["steps"].as_u64().unwrap() as usize;
    let mut s = scenario(&case, Model::Nonlinear(teacher));
    let ours = integrals(&mut s, steps, dt);
    let error = relative(s.height(), &f64s(&case["final"]));
    let ledger = ledger_error(&ours, &f64s(&case["integrals"]), mass_scale(s.height(), dx));
    let clipped: u64 = case["clippedFacesPerStep"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .sum();
    let report = s.diagnostics();
    println!(
        "teacher: field error {error:e}, integral error {ledger:e}, clipped faces {} (python {clipped})",
        report.limited_faces
    );
    assert!(error <= 1e-12);
    assert!(ledger <= 1e-14);
    assert!(clipped > 0, "the case should exercise the clip");
    assert_eq!(report.limited_faces, clipped);
    assert!(report.residual_relative.abs() <= 1e-14);
}

/// One arm against its torch rollouts.
///
/// Rust evaluates the network in f64 from the float32 weights, which is what
/// torch does after `.double()`, so the f64 rollout agrees to summation-order
/// rounding: 1e-9 of the largest height is a loose bound. The float32 torch
/// rollout keeps heights near 100 m in float32, whose spacing is 7.6e-6 m, and
/// rounds once per step and once per layer: after 16 steps it sits a few
/// 1e-7 of the largest height from the f64 result (the exporter records this
/// as `float32.relativeToMaxHeight`). 1e-5 leaves a factor of about 30.
fn learned_arm(arm: &str) {
    let case = load(&format!("arm-{arm}"));
    let closure = closure(arm);
    let (dt, dx) = (num(&case, "dtYears"), num(&case, "spacingM"));
    let steps = case["steps"].as_u64().unwrap() as usize;
    let start = f64s(&case["initial"]);
    let grid = Grid::new(case["side"].as_u64().unwrap() as usize, dx).unwrap();

    let mut tendency = vec![0.0; grid.cells()];
    closure.tendency(&grid, &start, &mut Work::default(), &mut tendency);
    let tendency_error = relative(&tendency, &f64s(&case["tendency64"]));

    let mut s = scenario(&case, Model::Learned(closure.clone()));
    let ours = integrals(&mut s, steps, dt);
    let scale = mass_scale(s.height(), dx);
    let error64 = relative(s.height(), &f64s(&case["final64"]));
    let ledger64 = ledger_error(&ours, &f64s(&case["integrals64"]), scale);
    let final32: Vec<f64> = f32s(&case["final32"]).into_iter().map(f64::from).collect();
    let error32 = relative(s.height(), &final32);
    let ledger32 = ledger_error(&ours, &f64s(&case["integrals32"]), scale);
    let initial = s.diagnostics().initial_integral_m3;
    let drift = ours.iter().fold(0.0f64, |m, v| m.max((v - initial).abs())) / scale;
    println!(
        "{arm}: tendency {tendency_error:e}; after {steps} steps of {dt} yr: field vs torch f64 {error64:e}, \
         vs torch f32 {error32:e}; integrals vs f64 {ledger64:e}, vs f32 {ledger32:e}; \
         integral drift {drift:e} of sum|h|a"
    );
    assert!(tendency_error <= 1e-10);
    assert!(error64 <= 1e-9);
    assert!(ledger64 <= 1e-12);
    assert!(error32 <= 1e-5);
    if matches!(
        closure.apply,
        landscape_core::Apply::Flux | landscape_core::Apply::Conductance { .. }
    ) {
        assert!(drift <= 1e-14, "the {arm} arm must conserve to rounding");
    }
}

#[test]
fn flux_arm_matches_torch_and_conserves() {
    learned_arm("flux");
}

#[test]
fn kfield_arm_matches_torch() {
    learned_arm("kfield");
}

#[test]
fn penalty_arm_matches_torch() {
    learned_arm("penalty");
}

#[test]
fn conductance_arm_matches_torch_and_conserves() {
    learned_arm("conductance");
}

#[test]
fn weights_meta_round_trips() {
    for arm in ["flux", "kfield", "penalty", "conductance"] {
        let c: Closure = closure(arm);
        assert_eq!(c.layers.len(), 5);
        assert_eq!(c.layers.last().unwrap().cout, 1);
        assert_eq!(c.spacing_m, 50.0);
    }
}
