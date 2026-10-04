# Methods

The question, hypotheses and decision rules are in [scientific-question.md](scientific-question.md) and
[protocol-v2.md](protocol-v2.md). This file describes how the products, models and measurements are built. The
v1 methods are kept in [methods-v1.md](methods-v1.md).

## Products and bytes

Every compared product is one `.gnc` file ([`codecs/package.py`](../src/geoneural/codecs/package.py)): a fixed
header (shape, lattice, bound, georeference, coder, model hash, table id), a component directory with lengths and
CRC32 checksums, and the components. A decoder needs this file and nothing else, except a shared model for corpus
products, which the header names by its SHA-256. The reader refuses unknown versions, flags, lattices, coders,
component kinds, damaged components, trailing bytes and streams that end early or leave bits unused.

Conventional codecs are wrapped in the same container (their raw stream as one component), so container overhead
is charged alike. The byte count of a product is the file length. Bytes per node use the 1025^2 unique nodes.

* **Standalone**: one field per file; a learned model is embedded (3,674 bytes for the development models, 3,678
  for the frozen ones, whose file format also records the size of an optional class embedding).
* **Corpus**: the same file without the model, which is stored once for all fields; the difference per field is
  the model plus its 9-byte directory entry.
* **Paged**: 257 x 257 pages with shared edges, coded alone, plus an 8-byte directory entry per page; duplicated
  edge nodes are charged.

## The multilevel coder

[`codecs/multilevel.py`](../src/geoneural/codecs/multilevel.py) codes a (2^k+1)^2 field under a maximum error e.

* **Lattice.** Heights are rounded to a 1 mm integer lattice. The bound becomes E = floor((e - 0.5 mm) / 1 mm)
  lattice units and the quantisation step q = 2E + 1, so every reconstructed node is within E mm of its lattice
  value and within e of the original height. E = 0 is lossless on the lattice.
* **Traversal.** As in SZ3's interpolation mode: the stride-64 sub-lattice is stored directly (zigzag deltas,
  zstd); then, for strides 64 down to 2, the new rows are predicted along columns from four known rows, and the
  remaining nodes along rows from four known columns. Predictions use only reconstructed values.
* **Residuals.** The integer error k = floor((Z - P + E) / q) is coded with rANS
  ([`codecs/rans.py`](../src/geoneural/codecs/rans.py)): a token (zero, or the sign and bit length of |k|) under one
  of 72 fixed tables, and the remaining bits raw. The tables are discretised Laplace distributions on a geometric
  scale grid, quantised to 15-bit integers and part of the format. Coding is within a few bytes of the tables'
  ideal length.
* **Exactness.** The decoder path uses only float32 add, multiply, divide and square root in a fixed order,
  integer arithmetic and rounding half to even. A log2 approximation reads the float32 bits. A separate Rust crate
  ([`native/gnc`](../native/gnc)) decodes the same files bit for bit, natively and as WebAssembly, on 28 parity
  fixtures and on a full 1025^2 field.

## Predictors

* **cubic-order0**: the cubic stencil (-1, 9, 9, -1) / 16 (quadratic or linear near the edges) and one rANS table
  per pass, chosen by the encoder.
* **cubic-ctx**: the same stencil; the table index per node is A + C log2(spread / q), where the spread is the
  standard deviation of the 4 x 3 stencil of reconstructed neighbours. A and C are fitted per pass by the encoder
  (grid search on the ideal code length) and stored. This is a conventional method and part of the conventional
  frontier.
* **learned**: an MLP (21 inputs, two hidden layers of 32 with hard-swish, 2 outputs) corrects the cubic prediction
  in units of the local spread and returns log2 of the symbol scale, which selects the table. Inputs: the 4 x 3
  stencil relative to the cubic prediction and divided by the spread, log2(spread / q), the level (one-hot), the
  pass and log2 of the bound in metres, so one model serves every bound. Weights are float16 in the model file
  ([`codecs/predictor.py`](../src/geoneural/codecs/predictor.py)).

**Training.** Closed loop: training samples are taken from real encodings of the training fields at all nine
bounds (40,000 per pass and field). Round 1 encodes with the cubic predictor, round 2 with the round-1 model, so the
features describe reconstructions a decoder really sees. Loss: code length under a discretised Laplace with
additive uniform noise standing in for rounding. 5,000 Adam steps per round, batch 65,536, cosine schedule. Models
for development results are trained leave one region out (the model for a region never sees it); the frozen
confirmation models are trained on all six development regions, seeds 0, 1 and 2.

## Conventional arms

SZ3 3.3.2, SPERR 0.8.5 and raw LERC through imagecodecs 2026.8.16, zfp through zfpy (fixed accuracy), q32 from
this package, each at its absolute-error setting. `sz3-best` runs the SZ3 3.3.2 command-line tool over 14
configurations (interpolation with Lorenzo, cubic, linear, other direction, Lorenzo with regression, first and
second order, and eight level-wise bound settings) and keeps the smallest stream that holds the bound
([`codecs/sz3tuned.py`](../src/geoneural/codecs/sz3tuned.py)). The best conventional product for a region and bound
is the smallest conventional product with no bound violation beyond one float32 ulp.

## Geology (H2)

GK100 units are rasterised by their INSPIRE material label onto a 40 m lattice with one class dictionary for all
regions (11 classes plus "no mapped unit"). A 4-dimensional class embedding (nearest coarse node) is appended to the
predictor's inputs. Every arm starts from the H1 model of the same fold and seed and gets one identical extra
closed-loop round (3,000 steps): no context, constant class, the real map, the map rolled by 2.56 km, 1 km blocks
shuffled, and the map of another region. The class raster of the real and control maps is stored in the product
(zstd) and charged; "map already at the decoder" is reported separately.

## Bound allocation (H3)

The bound may be tightened for the levels finer than stride 8, by a rule the decoder repeats on already decoded
values: near the drainage network routed on the decoded coarse lattice (factor 0.25 or 0.5 of E), or on gentle
slopes of the decoded coarse lattice (E scaled by (slope / 2%)^gamma, floored at 0.25 or 0.5). Allocated and uniform
bounds are compared at equal bytes by interpolating the uniform curve in log bytes.

## Process prior (H4)

Two matched sets of 48 synthetic 513 x 513 fields at 10 m ([`physics/synthetic.py`](../src/geoneural/physics/synthetic.py)):
landscapes evolved by the stream-power and diffusion teacher from noise, and procedural fields (power-law spectrum
plus ridged noise) matched to them in height standard deviation and median slope, with no routing or erosion.
Arms per fold: the H1 model (real only); the H1 model trained two more rounds on real data (same optimiser steps as
the pretrained arms, more real exposure); pretraining on process or procedural fields followed by the two real
rounds of H1; and the pretrained models without real data.

## Measurements and statistics

Height: maximum, RMSE, MAE, p99, bias, bound violations. Drainage
([`metrics/drainage.py`](../src/geoneural/metrics/drainage.py)): priority-flood filling, D8 by slope per metre,
accumulation; streams at a fixed contributing area of 0.05 km2 (0.025, 0.1 and 0.2 km2 as secondary thresholds);
exact Jaccard, recall and precision, a tolerant F1 that counts a stream cell as found when the other surface has
one within one cell, receiver agreement on interior cells, outlet agreement, and the changes of the deepest fill
(m) and the filled volume (m3). Every edge cell is an outlet. The reference noise floor (1 cm white noise on the
reference itself) is reported next to drainage results. Regions are the statistical unit: paired log ratios of bytes
per region, their geometric mean, a bootstrap over regions and the leave-one-region-out range.

## Physics (H5, H6)

* **Teacher.** Stream power with m = 0.5 and n = 1 on priority-flood filled D8 areas, five-point hillslope
  diffusion and uniform uplift, explicit Euler ([`physics/landscape.py`](../src/geoneural/physics/landscape.py)).
  It takes whole steps plus one remainder step and records the realised time; incision is limited by the actual
  drop to the receiver. Nondimensional groups are defined once in
  [`physics/units.py`](../src/geoneural/physics/units.py). The audit adds a non-divisible step, an analytic steady
  state, a manufactured diffusion solution, grid and step refinement, a rotated surface and a cross-check against
  fastscapelib 0.3.0.
* **Closures** ([`physics/hybrid.py`](../src/geoneural/physics/hybrid.py)): one flux per cell face. The accepted
  arm is a bounded symmetric conductance a = D_lin + (a_max - D_lin) sigmoid(NN(face features)), F_ij = a_ij
  (z_i - z_j), so a flat surface stays still, a constant offset changes nothing and the explicit step follows from
  a_max. Arms are selected by rollout error at 8, 32 and 64 steps in physical time over five seeds, not by
  one-step error.
* **Identifiability** ([`physics/inverse.py`](../src/geoneural/physics/inverse.py),
  [`physics/diagnostics.py`](../src/geoneural/physics/diagnostics.py)): observations at absolute epochs, noise from a
  declared exponential covariance, a Cholesky likelihood with an optional per-epoch datum, the scale c profiled
  along (cU, cK, cD, t/c) with timesteps rescaled consistently, and rank-normalised split R-hat with bulk and tail
  ESS for samplers.

## Reconstruction

The prototype coarsens 10 m terrain to 40 m with the node-centred trapezoidal operator, then compares bicubic
interpolation, bicubic with back-projection run to 1e-4 m, the best linear kernel fitted on training regions and a
U-Net on local, scale-normalised features computed on the whole field before tiling, trained on four regions and
tested on two others. Missing blocks are filled by harmonic and biharmonic solves or a U-Net from the surrounding
10 m data only. The operator label and fingerprint in [`superres`](../src/geoneural/superres) now describe and hash
the actual weights and code.
