# Results

State of the v2 study when it was paused on 4 October 2026. Each result says whether it is **confirmed** (frozen
recipe, run once on the seven untouched regions), **development** (six inspected regions, leave one region out),
**prototype** (quick exploratory runs) or **not finished**. Reports are in [`results/v2/`](../results/v2/); the
numbers shown in the case study are collected in [`results/v2/case-study.json`](../results/v2/case-study.json).
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

**Cost:** the learned decoder takes about 1.2 s per 1025^2 field in Python and about 0.7 s as WebAssembly in a
browser, against about 0.01 s for SZ3 (unqualified timings on a shared machine). Bytes are bought with decode time.

**Exactness:** the Rust decoder in [`native/gnc`](../native/gnc) reproduces the Python decoder bit for bit on 28
fixtures and on a full field with the trained model, natively and as WebAssembly.

## H2. Geology

**Prototype only; the full study was stopped.** In the prototype (two regions, two seeds) the real GK100 map saves
0.7 to 1.5% of the terrain stream, more than a shifted or a constant map, but storing the 40 m class raster costs
1.8 to 3.9 kB, more than it saves. The full study with six folds and all controls was running when the work was
paused; its models in progress were discarded.

## H3. Bound allocation for drainage

**Development, negative.** Tightening the bound near the drainage network the decoder routes on decoded coarse
levels, or on gentle slopes, never raises the tolerant stream F1 by the required 0.05 at equal bytes (mean change
-0.009 to +0.007 across rules and coders). The stream rule with factor 0.25 raises exact-cell Jaccard by about 0.05
but RMSE by 44 to 48%.

## H4. Process prior

**Not finished.** The synthetic sets exist (48 process-made and 48 matched procedural 513 x 513 fields, see
`physics/synthetic.py`); pretraining had started when the work was paused.

## H5. Learned closures

**Development.** Among learned arms with exact flat-state preservation and conservation, the bounded symmetric
conductance with a floor at the linear diffusivity has the lowest rollout error over five seeds (median 1.34 m,
against 3.45 m without the floor and 4.11 m for a learned face diffusivity). The free flux and penalty arms are
excluded: the free flux moves a flat surface, the penalty arm does not conserve material. With a piecewise material contrast of true ratio 0.2, the learned
ratio is 0.168 to 0.183. The teacher itself does not meet the spec's numerical budget at a 400-year step (its
discretisation error is 0.94 of the dynamics budget and 3.8 times the inverse noise level), so learned and
inverse results inherit that error; a 50-year step meets the dynamics budget.

## H6. Identifiability

**Development; coverage campaign not finished.** Scaling (U, K, D, t) to (cU, cK, cD, t/c) leaves one terminal
surface and two fractional epochs exactly unidentifiable when timesteps are rescaled consistently. A fixed
timestep cap breaks the symmetry numerically and invents 120 to 4,900 log-units of false information. Two surveys a
fixed number of years apart make c identifiable for transient truths (prototype: 2 log-unit interval 0.97 to 1.02
with a 50 kyr gap) and not near equilibrium. In the prototype the correct correlated likelihood covered the truth in
95 to 97% of 300 noise draws, the old independent sum of squares in 37 to 50%. The repeated-truth coverage study,
the CNN and feature estimators and the misspecification runs were stopped before finishing.

## Reconstruction

**Prototype; the full study was stopped.** From 40 m to 10 m on held-out Essen and Rothaar, a U-Net lowers MAE by
17 to 20% against bicubic interpolation with back-projection to 1e-4 m; the best linear kernel only ties bicubic,
and the gain disappears when the observation is point-decimated instead of averaged (the model learns the
averaging kernel). Missing 64 x 64 blocks: a modest height gain over biharmonic infill and no drainage gain.

## What changed from v1

* The v1 comparator (q32) needs 2 to 4 times the bytes of SZ3, so every v1 neural comparison used a weak bar.
* Fitting a coordinate network to one field still does not pay: on Essen at 0.5 m it saves 4.4 kB of residual for
  29 kB of latents and weights.
* Sparse corrections are now written as a real stream with a 17-byte header (magic, cell count, quantum, coder),
  so each correction product is 17 bytes larger than in the v1 tables; nothing else about them changes.
* The v1 drainage finding stands in a sharper form: at a fixed maximum error, the codec that keeps the lowest RMSE
  (SPERR) keeps the most streams, whatever its bytes.
