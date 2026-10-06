# Results

State of the v2 study on 6 October 2026. Each result says whether it is **confirmed** (frozen recipe, run once on
untouched regions), **development** (six inspected regions, leave one region out), or **prototype** (quick exploratory
runs). There are two confirmation cohorts: cohort A (seven regions, used for H1 and the
reconstruction runs) and cohort B (seven more regions, drawn by the same ranking after H1b was frozen). Reports are
in [`results/v2/`](../results/v2/); the numbers shown in the case study are collected in
[`results/v2/case-study.json`](../results/v2/case-study.json).
The v1 results are kept in [results-v1.md](results-v1.md).

## H1. A learned predictor and complete bytes

**Confirmed on bytes, not supported under the frozen rule.** At the same maximum error, every node checked, the
learned multilevel coder with its model embedded in each file is smaller than the best conventional product (the
smallest of SZ3, the best of 14 SZ3 configurations, SPERR, zfp, LERC and q32) in all seven confirmation regions:

| Bound | Learned / best conventional (95% bootstrap over regions) | Learned / cubic-ctx | cubic-ctx / best conventional | Worst tolerant-F1 gap |
|---|---|---|---|---|
| 0.05 m | 0.826 (0.812 to 0.841) | 0.891 | 0.928 | -0.030 |
| 0.1 m | 0.795 (0.783 to 0.807) | 0.875 | 0.908 | -0.023 |
| 0.25 m | 0.744 (0.738 to 0.750) | 0.854 | 0.871 | -0.052 |
| 0.5 m | 0.771 (0.753 to 0.791) | 0.882 | 0.875 | -0.084 |
| 1 m | 0.940 (0.845 to 1.071) | 1.062 | 0.885 | -0.072 |

The ratios are geometric means over regions; per region the learned product is 14 to 27% smaller at 0.05 to
0.5 m. The rule in [protocol-v2.md](protocol-v2.md) also requires that the tolerant stream F1 drops by at most
0.01 against the best conventional product in every region. It does not hold. At 0.05 m three regions fail (two
against SPERR, on moderate and rough terrain), at 0.1 m one, at 0.25 m four and at 0.5 m all seven, mostly
against SZ3 configurations, with the largest drop on a flat region. The learned coder fills the bound more evenly:
at 0.5 m its RMSE is 20 to 32% higher than that of the best conventional product at the same maximum error, and
the stream overlap follows RMSE. So H1 as frozen is **not supported**, while its byte half is confirmed with tight
intervals at 0.05 to 0.5 m. About half of the gain is the coder (cubic prediction with a context-dependent
entropy model, on average 7 to 13% below the best conventional product); the learned predictor adds 11 to 15% on
average. At 1 m the byte half fails too: the embedded 3.7 kB model makes single files larger in two regions; with
the model shared (corpus product) the learned coder stays smaller (0.776 at 1 m).

Development results (six regions, three seeds, leave one region out) agree: 0.854 at 0.05 m to 0.753 at 0.5 m,
every region smaller, seeds within 2%. The drainage condition failed there too, at 0.1 to 0.5 m.

**Paged product (development):** 257 x 257 pages coded alone, model shared: the learned coder is 24 to 34% smaller
than SZ3 and SPERR pages at 0.1 to 1 m.

**Cost:** the learned decoder takes 0.46 s per 1025^2 field in Python on one thread (qualified, see
[Cost and scaling](#cost-and-scaling)) and about 0.7 s as WebAssembly in a browser (unqualified), against 0.009 s
for SZ3. Bytes are bought with decode time.

**Exactness:** the Rust decoder in [`native/gnc`](../native/gnc) reproduces the Python decoder bit for bit on 28
fixtures and on a full field with the trained model, natively and as WebAssembly.

## H1b. Level-wise bounds for the drainage condition

**Confirmed at 0.05 to 0.25 m, not supported over the full range.** H1 failed on drainage because, at the same
maximum error, the learned coder spreads its error more evenly than SZ3 and so has a higher RMSE. H1b codes the
coarse levels to a tighter bound (the largest error is unchanged), so the nodes every later prediction is built from
stay closer to the truth. In development ([`levels.json`](../results/v2/codec/levels.json), nine rules, the rule
chosen per region on the other five) the H1 rule passed at 0.05 to 0.5 m. The rule per bound and the frozen H1
models were then frozen ([`levels-recipe.json`](../results/v2/frozen/levels-recipe.json)) and run once on cohort B
([`h1b-levels-confirmation.json`](../results/v2/codec/h1b-levels-confirmation.json)):

| Bound | Learned / best conventional (95% bootstrap) | All regions smaller | Worst tolerant-F1 gap | H1 rule |
|---|---|---|---|---|
| 0.05 m | 0.888 (0.882 to 0.895) | yes | -0.004 | pass |
| 0.1 m | 0.871 (0.854 to 0.889) | yes | +0.002 | pass |
| 0.25 m | 0.871 (0.844 to 0.901) | yes | +0.001 | pass |
| 0.5 m | 0.985 (0.905 to 1.076) | no | +0.009 | fail (bytes) |
| 1 m | 1.059 (0.907 to 1.256) | no | -0.011 | fail |

From 0.05 to 0.25 m the variant meets every condition of the frozen rule on untouched regions, the stream condition
included. At 0.5 and 1 m the tighter coarse levels cost the byte margin. The RMSE of the learned products is
0.64 to 0.98 of the uniform cubic-ctx coder's.

## Drainage: stream threshold, outlets and routing domain

**Re-analysis of both confirmation runs, plus development.** The same products as in the H1 and H1b confirmations
(frozen models, the best conventional product each run recorded) scored at four stream thresholds and under two
outlet policies: every edge cell may drain out (frozen), or water leaves only through the lowest edge cell
([`drainage-sensitivity.json`](../results/v2/codec/drainage-sensitivity.json)). At the frozen setting the numbers
reproduce both confirmation reports exactly.

H1 (uniform bound, cohort A) fails at every threshold and bound with every edge as an outlet, and passes only at
0.1 m when water leaves through the lowest edge cell. Its drainage failure does not depend on these choices.

H1b (level-wise bound, cohort B), worst region's tolerant F1 minus the best conventional product's:

| Contributing area | Outlets | 0.05 m | 0.1 m | 0.25 m | 0.5 m | 1 m |
|---|---|---|---|---|---|---|
| 25,000 m² | every edge | -0.001 | +0.008 | -0.001 | +0.004 | -0.010 |
| 50,000 m² (frozen) | every edge | -0.004 | +0.002 | +0.001 | +0.009 | -0.011 |
| 100,000 m² | every edge | -0.015 | +0.001 | +0.002 | +0.013 | -0.010 |
| 200,000 m² | every edge | -0.015 | -0.018 | -0.015 | -0.006 | -0.011 |
| 25,000 m² | lowest edge | +0.001 | +0.005 | -0.001 | +0.005 | -0.001 |
| 50,000 m² | lowest edge | -0.002 | +0.003 | -0.004 | +0.007 | +0.003 |
| 100,000 m² | lowest edge | -0.016 | +0.004 | -0.008 | +0.020 | +0.007 |
| 200,000 m² | lowest edge | -0.021 | -0.014 | -0.009 | +0.012 | -0.008 |

The rule allows -0.010. At the frozen threshold and at the denser 25,000 m² network, H1b passes at 0.05 to 0.25 m
under both outlet policies. With fewer, larger streams single regions fall up to 2.1 points behind, mostly at
0.05 m. Averaged over regions, H1b keeps more streams than the best conventional product in every cell of the table.

Routing domain (development): each region was also routed inside a domain 5.12 km wider on every side. The
reference streams themselves change: the tolerant F1 between the region's streams routed alone and routed in the
wider domain is 0.97 to 0.99 on the hilly regions and 0.88 to 0.90 on the flat Lower Rhine and Münsterland regions
(50,000 m²). That is larger than any codec effect in this study. The codec comparison barely moves: the
worst-region gap of the learned coder changes by at most 1.4 points between the two domains (seed 0), and the
level-wise variant passes or fails at the same bounds in both.

## Conventional arms C1 to C3

**Development.** Six regions, every product decoded from its own bytes
([`frontier.json`](../results/v2/codec/frontier.json)). These arms ask which part of the learned coder carries the
gain, using only conventional parts.

| Bound | C1 / best conventional (95% bootstrap) | C1 / cubic-ctx | C1 / learned | C3 morphology / C1 | C3 geology / C1 (shifted map) | C2: stream F1 at equal bytes, mean (worst region) |
|---|---|---|---|---|---|---|
| 0.05 m | 1.003 (0.999 to 1.008) | 1.079 | 1.175 | 1.002 | 1.011 (1.010) | +0.006 (-0.003) |
| 0.1 m | 1.007 (0.996 to 1.022) | 1.096 | 1.217 | 1.001 | 1.012 (1.012) | +0.023 (+0.001) |
| 0.25 m | 1.001 (0.993 to 1.010) | 1.137 | 1.300 | 0.993 | 1.015 (1.017) | +0.021 (-0.000) |
| 0.5 m | 1.008 (0.978 to 1.037) | 1.177 | 1.339 | 0.999 | 1.023 (1.028) | +0.023 (-0.003) |
| 1 m | 1.014 (0.987 to 1.043) | 1.187 | 1.238 | 0.972 | 1.049 (1.048) | +0.027 (-0.001) |

* **C1, coarse base plus residual.** A coarse lattice (every 2nd to 16th node) coded by SZ3, cubic interpolation,
  and the residual coded by SZ3 or SPERR at the target bound; 24 settings, chosen per region on the other five. It ties the best conventional product and is 18 to 34% larger than the learned coder. Splitting the
  field into a base and a residual does not explain the learned coder's margin; the gain comes from predicting each
  level from the decoded coarser levels and coding the residual with a context-dependent entropy model.
* **C3, regression on the base.** A ridge regression of the C1 residual on slope, curvature and their product, with
  the coefficients in the file, saves up to 2.8% at 1 m and nothing at 0.05 m. With GK100 class indicators added
  (the 40 m class raster charged), the products are 1 to 5% larger, the same as with the map shifted by 1 km: the
  classes explain no part of the residual.
* **C2, sparse corrections.** A uniform SZ3 or SPERR product plus corrections to a quarter of the bound on the
  stream cells (band 0), the stream cells and their neighbours (band 1), or cells whose flow direction could turn
  within the bound. Compared with a uniform product of the same total bytes (a tighter bound, interpolated over 13
  bounds), the best rule (SZ3, band 0) adds 0.6 to 2.7 points of tolerant stream F1 on average, with the worst region
  within 0.3 points of no change. That is below the 5 points the protocol asks of an allocation rule (H3). The
  flow-direction rule adds nothing on average.
* **Portfolio.** Taking the smaller of the learned and the best conventional product per field changes nothing at
  0.05 to 0.5 m (the learned product is smaller in every region) and, at 1 m, picks the conventional product for the
  flat Münsterland region only (0.3% on average).

All 1,146 products meet their bound within one float32 unit in the last place; 23 exceed it by at most 0.15 µm, the
rounding of the float32 residual.

## H2. Geology

**Development, negative.** Six folds, three seeds, six arms per fold ([`h2.json`](../results/v2/codec/h2.json)).
Every arm starts from the same H1 model and gets the same extra training; only the context differs. With the 40 m
GK100 raster charged, the real map makes standalone products 0.9% (0.05 m) to 10% (1 m) larger than the matched
no-context model. With the map taken as already at the decoder, the real map is never better than a shifted or block-shuffled copy
of itself, which keep the class statistics but break the link to the terrain; all context arms stay within 0.6% of
each other. The coder finds no usable geological signal. The prototype's 0.7 to 1.5% saving (two regions, two seeds) did not hold up. The rule fails at every bound,
so no recipe was taken to confirmation.

Scale and position of the map ([`geology-sweep.json`](../results/v2/codec/geology-sweep.json)): from the same H2
recipe and seed, eight more arms use a 10 m or a 160 m class raster instead of 40 m, the map shifted by 120 m, 320 m
or 1 km, class borders blurred over 3 or 5 cells, or a quarter of the 1 km blocks set to unknown. With the map at the
decoder, every arm is within 0.5% of the real 40 m map at every bound from 0.5 mm to 2 m, and none is smaller than
the no-map model by more than 0.12%. A map shifted by 1 km gives the same bytes as the real one. Charged, the 10 m
map makes products 1 to 73% larger and the 160 m map 0.02 to 5% larger than no map. No resolution pays for itself.

## H3. Bound allocation for drainage

**Development, negative.** Tightening the bound near the drainage network the decoder routes on decoded coarse
levels, or on gentle slopes, never raises the tolerant stream F1 by the required 0.05 at equal bytes (mean change
-0.009 to +0.007 across rules and coders). The stream rule with factor 0.25 raises exact-cell Jaccard by about 0.05
but RMSE by 44 to 48%.

## H4. Process prior

**Development, small gain, rule met at 1 m only.** Six folds, three seeds ([`h4.json`](../results/v2/codec/h4.json)).
Pretraining the shared predictor on 48 simulated landscapes before the two real-data rounds gives fewer bytes than
both controls with equal optimiser steps, and the bootstrap interval excludes no change at every bound. The gain is
0.2 to 0.9% against extra real training and 0.7 to 1.3% against a procedural (non-physical) prior from 0.05 to
0.5 m, 2.9% and 2.3% at 1 m. The rule asks for 2% against both, so it passes at 1 m only. A model trained on
simulated terrain alone is 17 to 20% worse than the real-data model: the simulation helps as a starting point, not
as a substitute.

## H5. Learned closures

**Development.** Among learned arms with exact flat-state preservation and conservation, the bounded symmetric
conductance with a floor at the linear diffusivity has the lowest rollout error over five seeds (median 1.34 m,
against 3.45 m without the floor and 4.11 m for a learned face diffusivity). The free flux and penalty arms are
excluded: the free flux moves a flat surface, the penalty arm does not conserve material. The browser lab runs the
median seed (seed 5, retrained with the same recipe, 1.34 m on the same rollouts) in the Rust kernel, which matches
the Python arm to rounding. With a piecewise material contrast of true ratio 0.2, the learned
ratio is 0.168 to 0.183. The teacher itself does not meet the spec's numerical budget at a 400-year step (its
discretisation error is 0.94 of the dynamics budget and 3.8 times the inverse noise level), so learned and
inverse results inherit that error; a 50-year step meets the dynamics budget.

## H6. Identifiability

**Development.** Scaling (U, K, D, t) to (cU, cK, cD, t/c) leaves one terminal
surface and two fractional epochs exactly unidentifiable when timesteps are rescaled consistently. A fixed
timestep cap breaks the symmetry numerically and invents 120 to 4,900 log-units of false information. Two surveys a
fixed number of years apart make c identifiable for transient truths (prototype: 2 log-unit interval 0.97 to 1.02
with a 50 kyr gap) and not near equilibrium. In the prototype the correct correlated likelihood covered the truth in
95 to 97% of 300 noise draws, the old independent sum of squares in 37 to 50%.

Repeated truths ([`inverse-coverage.json`](../results/v2/physics/inverse-coverage.json)): with the correct model,
the 95% intervals of the fixed-lag design cover the truth in 90% of 100 one-ratio truths and 30 two-ratio truths,
at the lower edge of the protocol's 0.90 to 0.98; the 68% intervals cover 58 to 80%, and a Laplace approximation
only 54 to 74%. Under misspecification ([`inverse-misspecification.json`](../results/v2/physics/inverse-misspecification.json))
the narrow intervals become overconfident: 60% coverage with a finer simulation step, 10% with a 10% uplift
gradient, 5% with shifted boundary values and 0% with the FastScape solver or a different initial surface. A CNN
([`inverse-estimators.json`](../results/v2/physics/inverse-estimators.json)) gives calibrated intervals but 3.6 to
5.3 times the error of the likelihood reference; a quadratic feature regressor also misses the target of at most
twice the reference error.

Nuisance ratios ([`inverse-nuisance-profile.json`](../results/v2/physics/inverse-nuisance-profile.json)): with the
uplift and diffusivity ratios profiled at every c (offsets up to 0.03 log units) instead of fixed at the truth, the
2 log-unit interval for c with a 50 kyr lag stays 0.985 to 1.014 (expected data, transient truth). The best offsets
stay within 0.0033 log units of the truth, so the other ratios do not absorb the scale factor.

## Reconstruction

**Confirmed for coarse-to-fine and sparse samples; missing blocks negative.** Each task was developed with leave one
region out on the six development regions (three seeds, the training length chosen on a validation region), its
recipe frozen by the development report, and then run once on cohort A
([`results/v2/recon/`](../results/v2/recon/)). The protocol asks for at least 10% lower MAE than the best non-neural
method with identical inputs, observation consistency and no lower tolerant stream F1.

| Task | Development: neural against the best non-neural | Cohort A | Rule |
|---|---|---|---|
| 40 m averages to 10 m | 0.188 m against 0.217 m (regression kriging), 13.2% lower | 0.116 m against 0.139 m, 16.2% lower; F1 +0.028 | confirmed |
| 1% noisy samples | 1.098 m against 1.235 m (kriging), 11.1% lower | 0.511 m against 0.593 m, 13.9% lower; F1 0.292 against 0.269 | confirmed |
| 5% noisy samples | 0.415 m against 0.473 m, 12.3% lower | 0.271 m against 0.311 m, 13.1% lower; F1 0.416 against 0.385 | confirmed |
| 8-cell holes | 0.259 m against 0.278 m (biharmonic), 7.0% lower | 0.164 m against 0.174 m, 5.6% lower | not met |
| 32-cell holes | 2.4% lower | 1.7% lower | not met |
| 128-cell holes | 0.4% lower | 0.3% lower | not met |
| 10 m to 1 m (three regions with 1 m data) | 0.089 m against 0.096 m (regression kriging), 7.5% lower; F1 +0.037 | no 1 m data | not met |

Coarse-to-fine estimates are back-projected so that they reproduce every observed 40 m average to 1e-4 m; the
five-member ensemble's 90% intervals cover 92 to 93% of nodes. Geology as an extra input lowered the coarse-task MAE
by 3 mm in development and not at all on cohort A. With noisy samples, observation consistency is not imposed (the
samples carry declared noise). The 10 m to 1 m task is development only, since no confirmation region has 1 m data.

Ambiguity ([`ambiguity.json`](../results/v2/recon/ambiguity.json), models of two development folds): pairs of
fields with identical 40 m averages. In one pair a 2 m deep channel on a planar slope moves two cells sideways; in
the other, a region's 10 m detail is swapped for another region's. No method can tell the two apart, so the 90%
intervals should contain both answers wherever they differ by more than 0.5 m. The five-member ensemble
contains both at 50 to 93% of those nodes, a single model at 5 to 71% and regression kriging at 0 to 22%. The
ensemble widens its intervals where the answer is open, but not enough to reach the nominal 90%.

## Cost and scaling

**Measured on an idle host** ([`results/v2/bench/`](../results/v2/bench/)): Intel i9-13900K, one thread per process
unless stated, Essen at 0.25 m. A timing counts as qualified when the host was less than 10% busy before and after
it; 29 of 30 query timings and every profile and scaling timing qualify.

* **Encode and decode.** SZ3 encodes in 0.02 s and decodes in 0.009 s, zfp decodes in 0.011 s, SPERR in 0.023 s.
  The learned coder encodes in 0.52 s and decodes in 0.46 s, 50 times SZ3. Almost all of that is the level-by-level
  prediction loop; the entropy decoder takes 9 ms. A fresh process spends 0.58 s on imports for every codec, and the
  learned decoder's first call another 0.43 s to load its compiled code. Routing one field for the drainage metric
  takes 0.65 s.
* **Queries.** The same arrays for every product: 1,000 random points, 1,000 clustered points, 16 windows of
  128 x 128 nodes, the full raster and every eighth node, each answer checked against the full decode. A 1025^2
  field has only 16 pages of 257^2 nodes, and the point and window queries touch 13 to 16 of them, so the paged
  learned product saves at most 16% of the decode time (0.39 s against 0.46 s for clustered points) for 3.4% more
  bytes. Paging pays on extents larger than one region.
* **Training.** One shared model with the frozen recipe (six regions, all bounds, two rounds of 5,000 steps) takes
  58 s: 38 s to collect 12 million samples on one CPU thread and 20 s of optimisation on an RTX 4070 Ti (the GPU
  showed 11 to 28% load from other processes).
* **Amortisation** ([`amortisation.json`](../results/v2/codec/amortisation.json)). The embedded 3.7 kB model pays
  for itself within one field from 0.5 mm to 0.5 m in every cohort A region, and within 1 to 8 fields at 1 m. At
  2 m it never pays in three of seven regions: there the learned product is larger than the best conventional one
  even without the model.
* **Scaling.** Independent region encodes as a job array (26 tasks of about 0.5 s, three repeats, median wall
  time). Strong scaling efficiency is 0.98, 0.90, 0.74 and 0.53 at 2, 4, 8 and 16 workers; weak scaling (four tasks
  per worker) 1.00, 0.99, 0.85 and 0.61. The loss comes from slower tasks, not from coordination: a task takes
  0.51 s alone and 0.81 s with 16 workers running. The CPU has 8 performance and 16 efficiency cores and lowers its
  clock under load.

## Can the headline claim be made?

"Geology- and physics-informed neural networks beat conventional terrain compression" needs every condition below.

| Condition | Holds? | Evidence |
|---|---|---|
| The complete neural product is smaller or more accurate under a frozen, meaningful rule | yes, at 0.05 to 0.25 m | H1b on cohort B: 11 to 13% fewer bytes, streams kept; H1 met its byte half only |
| Strong conventional codecs with the same access and metadata | yes | SZ3 (default and best of 14), SPERR, zfp, LERC, q32 and C1 to C3, all in the same container; standalone and paged products |
| Decoded bytes, not live tensors or entropy estimates, produced the scored field | yes | every product decoded from its file; the Rust decoder matches Python bit for bit |
| Geology, shared models, boundaries, indexes and residuals have owners and costs | yes | models and maps charged in standalone files, 8 bytes per page, a 17-byte correction header; routing domain and outlets tested |
| The effect survives matched no-geology and no-process controls | geology and process: no | H2 and the geology sweep: no signal; H4: below the 2% bar except at 1 m |
| Independent geographic confirmation | yes, for H1 and H1b | two cohorts of seven regions drawn by a committed rule; geology and the process prior never reached confirmation |
| Elevation, drainage and boundaries meet the declared requirements | at 0.05 to 0.25 m, at the frozen stream density | holds for denser networks and both outlet policies; single regions fall up to 2 points behind for sparse networks |
| Numerical precision and runtime support the error contract | yes | integer 1 mm lattice, the bound checked on every node, float32 inputs within one unit in the last place |
| Encoding, training cost and amortisation are reported | yes | see [Cost and scaling](#cost-and-scaling) |
| The source of the benefit is named | yes | the multilevel prediction with a context-dependent entropy model; not geology, barely morphology (C3, at most 3%), a small process-prior gain (H4), no format-selection gain (portfolio) |

The headline claim fails on the fifth condition. What the evidence supports is narrower: a 3.7 kB learned
predictor inside a multilevel error-bounded coder stores 10 m NRW terrain with 11 to 13% fewer complete bytes than
the best of strong conventional codecs at a largest error of 5 to 25 cm, on untouched regions, without losing
streams. Geology and landscape physics add nothing measurable to that. The process prior shows a small, consistent
gain that does not reach the bar the protocol set.

## What changed from v1

* The v1 comparator (q32) needs 2 to 4 times the bytes of SZ3, so every v1 neural comparison used a weak bar.
* Fitting a coordinate network to one field still does not pay: on Essen at 0.5 m it saves 4.4 kB of residual for
  29 kB of latents and weights.
* Sparse corrections are now written as a real stream with a 17-byte header (magic, cell count, quantum, coder),
  so each correction product is 17 bytes larger than in the v1 tables; nothing else about them changes.
* The v1 drainage finding stands in a sharper form: at a fixed maximum error, the codec that keeps the lowest RMSE
  (SPERR) keeps the most streams, whatever its bytes.
