# Protocol v2

This file fixes how results are produced and judged. It was fixed before the confirmation regions were
downloaded. The record of the order is written into the files themselves: the cohort draw
(`results/v2/cohort.json`, `createdUtc`), the recipe freeze with the model hashes
(`results/v2/frozen/h1-recipe.json`, `frozenUtc`) and the confirmation report (`createdUtc`). Commit dates in this
repository are not the record, and the `sourceRevision` of a report names the working revision at run time, which
need not be on `main`. Changes after the freeze are listed at the end with their date and reason, and any result
they affect is labelled.

## Data

**Development regions (six, all inspected during development):** essen-ruhr, muensterland-plain, lower-rhine,
teutoburg-forest, bergisches-land, rothaar-sauerland. Each is a 1025 x 1025 node lattice at 10 m (10.24 km square)
of NRW DGM1 bare-earth terrain, EPSG:25832 with DHHN2016 heights (EPSG:7837). Three also have 1 m data. GK100
geology (1:100,000 surface units, material label) exists for all but teutoburg-forest, whose WFS response could
not be parsed. These regions are development evidence only.

**Confirmation regions (Tier C):** seven new 10.24 km tiles chosen by the rule in
[`src/geoneural/data/cohort.py`](../src/geoneural/data/cohort.py): a fixed 11 km candidate lattice over NRW, a
15 km buffer around all earlier regions, ranking by a SHA-256 of a fixed salt and the tile origin, one coarse
400 m probe per candidate in rank order (skip any with missing values), and relief strata from the coarse height
standard deviation (flat below 8 m: 2 tiles; moderate 8 to 30 m: 3; rough above 30 m: 2). The probe statistics
are the only look at these tiles before the frozen runs; every probed candidate is in the exposure log.

**Integer lattice.** Codecs of the multilevel family store heights on a 1 mm lattice. A bound e in metres becomes
E = floor((e - 0.5 mm) / 1 mm) lattice units, so the stored product guarantees |error| <= E mm + 0.5 mm <= e. The
1 mm lattice is a storage convention, not a claim about survey accuracy.

## Products and the byte ledger

* **Standalone raster** (primary): one 1025 x 1025 field in one `.gnc` file. Every component a decoder needs is
  inside: header, coarse lattice, per-pass parameters, rANS stream, raw offset bits, and for the learned coder its
  model.
* **Corpus**: the same, with the learned model stored once for all fields and referenced by hash. Reported next
  to the standalone number, with the model size and the break-even corpus size.
* **Paged** (secondary): 257 x 257 pages with shared edges, each decodable alone; duplicated edges are charged.

Conventional codecs are wrapped in the same container, so container overhead is charged equally. Bytes per node
use the 1025^2 unique nodes. Generic decoder software is not counted for any codec; the rANS tables are part of
the format and are not counted either.

## Arms

| Arm | Meaning |
|---|---|
| sz3, sperr, zfp, lerc, q32dz | Conventional codecs at their absolute-error setting (imagecodecs and zfpy versions recorded per run) |
| cubic-order0 | Multilevel coder, fixed cubic prediction, one rANS table per pass |
| cubic-ctx | Multilevel coder, fixed cubic prediction, table chosen per node from the local spread by a fitted two-parameter rule |
| learned | Multilevel coder with the shared learned predictor (mean and table) |
| learned + geology | The same with a GK100 class embedding as input; the class raster is charged |
| geology controls | Constant class, map shifted by 2.56 km, block-shuffled map, wrong-region map; same input width |
| allocation | Bound tightened near the drainage network that the decoder derives from already decoded coarse levels |
| process prior | Predictor pretrained on simulated landscapes, against matched extra real training and a procedural non-physical prior |

The best conventional product for a region and bound is the smallest conventional product that keeps the bound on
every node (one float32 ulp of rounding allowed and recorded). The cubic-ctx arm is a conventional method too:
neural gains are reported against it and against the best of the external codecs.

## Bounds and metrics

Bounds: 0.0005 (lossless on the lattice), 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1 and 2 m. The primary range for
decisions is 0.05 to 1 m.

Height: maximum, RMSE, MAE, p99, bias, bound violations (strict and ulp-tolerant) and largest excess. Drainage
(`metrics/drainage.py`): streams at a fixed 0.05 km2 contributing area (secondary 0.025, 0.1, 0.2 km2), exact
Jaccard, recall, precision, 1-cell tolerant F1, receiver agreement on interior cells, outlet agreement, change of
the deepest fill (m) and of the filled volume (m3). Every edge cell is an outlet. The reference noise floor (1 cm
white noise on the reference) is reported with every drainage result. Encode and decode times are measured on a
shared machine and are labelled unqualified unless the host was verified idle.

## Splits and statistics

Learned models are trained with leave one region out: the model for a region never sees it. Seeds (three for
development, the frozen seed set for confirmation) are optimisation variation, not independent evidence. The
statistical unit is the region: per bound, the paired log ratio of bytes per region, its geometric mean over
regions, a percentile bootstrap over regions and the leave-one-region-out range. Pixels and patches are never
treated as independent.

## Decision rules (frozen)

* **H1 (compression):** at each bound in 0.05 to 1 m, the learned standalone product is at least 10% smaller than
  the best conventional product (geometric mean over regions), the upper end of the bootstrap interval is below
  0.95, it is smaller in every confirmation region, and the tolerant stream F1 is not lower by more than 0.01. The
  same test against cubic-ctx decides whether the gain is the learned part or the coder.
* **H2 (geology):** total bytes including the charged map at least 1% below the matched no-context model, and the
  gain absent (within seed spread) for the shifted and shuffled maps. The case of a map already at the decoder is
  reported separately as a conditional result.
* **H3 (allocation):** at equal bytes (within 2%), tolerant stream F1 at least 0.05 higher than the uniform bound,
  with RMSE no worse.
* **H4 (process prior):** at least 2% fewer bytes than both the matched extra-real-training control and the
  procedural prior, with the bootstrap interval excluding no change.
* **H5 (closure):** the bounded conductance arm has the lowest 64-step rollout error among the learned arms over 5
  seeds while flat-state preservation and conservation hold to rounding.
* **H6 (inverse):** the fixed-lag design gives 95% intervals with coverage between 0.90 and 0.98 over repeated
  truths, and the fractional design stays flat within numerical error.
* **Reconstruction:** on held-out regions the neural model lowers MAE by at least 10% against the best non-neural
  baseline with identical inputs, keeps observation consistency, and does not lower the tolerant stream F1.

A result that misses a threshold is reported as negative or inconclusive, never as "almost confirmed". Gains below
a threshold but with an interval excluding zero are reported as small effects.

## Confirmation

After this file is fixed: select and download the cohort and prepare it without looking at codec or model
outputs. Development work continues on the six development regions only. Before the confirmation run, a recipe
file freezes the recipes: for each arm the training recipe, the seeds (0, 1, 2), and the hashes of the model
files, which are trained on all six development regions. Then every frozen arm runs once on the cohort; for
learned arms the result is the mean over the three seeds. The results are reported whatever they show. A redesign
after seeing them needs a new cohort.

## Changes after the freeze

* 2026-10-04: the GK100 download for teutoburg-forest failed because the provider announced features on its second
  page that the page did not contain. Fetched again in one page of 1000 features; all six development regions
  now have geology.
* 2026-10-04: a stronger conventional arm, `sz3-best` (the smallest of 14 SZ3 3.3.2 configurations per field and
  bound, `codecs/sz3tuned.py`), joins the conventional set. It can only make a neural gain harder to show.
* 2026-10-04: the decoder now refuses unknown flag bits, a lattice other than 1 mm, oversized products, and
  streams that end early or leave raw bits unused. No result changes; the Rust decoder applies the same checks.
* 2026-10-04: confirmation cohort selected by the committed rule: nrw-368-5747, nrw-390-5780, nrw-324-5648,
  nrw-313-5659, nrw-401-5791, nrw-445-5703, nrw-412-5769 (`results/v2/cohort.json`, with the exposure log of all
  30 probed candidates). H1 recipe and model hashes frozen in `results/v2/frozen/h1-recipe.json`.
