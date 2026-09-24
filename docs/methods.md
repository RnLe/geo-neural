# Methods

## The question

How small and fast can a terrain representation get before it stops being useful? "Useful" is
measured three ways: height error against the reference, the drainage network a routing algorithm
derives from the surface, and the cost of reading the representation back.

Two experiments that look alike are kept apart:

* **Codec fit.** The encoder sees every node of the field it compresses. Training a network on all
  nodes is the encoding step, and the result is judged like any other codec: bytes against
  reconstruction error on the same field.
* **Held-out prediction.** Pages withheld from training are scored. Normalisation statistics then
  come from the training pages only, because a withheld page must not reach the model even
  through a mean. A held-out score says something about generalisation; a codec fit does not.

## Counting bytes

A representation is charged for everything its decoder needs. Two package conventions exist and
never share an axis.

* `finest-per-page`: the finest lattice in independently compressed 65 x 65 pages. This is the
  product that matches a neural field, which answers any coordinate without reading neighbours.
  A coarse grid is charged the same way, at its own resolution. A hybrid network is charged its
  weights plus its conventional base at this convention. Quantised weights are entropy coded and
  their scales and tables are counted.
* `pyramid`: all five pyramid levels plus the JSON page index, as the atlas ships. The index alone
  is 220,815 bytes, about a third of the package at the 1 m target.

Bytes are decimal. Bits per sample divide by the 1,050,625 finest nodes unless stated.

## Errors

Mean, RMS, 99th percentile and maximum absolute error are taken over all 1,050,625 reference
nodes. A maximum is either a guarantee (a quantiser at step 2t meets max error t by construction)
or an observation (a coarse grid or a network has no bound on what it discards). The candidate
table records which one each row has.

## Comparing candidates

For one dataset and package convention, each candidate has measurements
(bytes, mean error, maximum error). A row dominates another only if it is no worse on every axis
and strictly better on at least one. One comparator row has to satisfy all inequalities: two
different conventional rows that each win one axis do not jointly dominate a candidate.

Dominance is yes or no, so the table also gives each learned row its best conventional comparator
(lowest mean error at no more bytes) and the margins between them. The conventional side is swept
densely: grids from the 10 m lattice down to 640 m (the five pyramid levels and two coarser
decimations), each at 19 error targets from 0.01 m to 16 m, plus the trivial control of a stored
field mean. A neural row that looks
undominated against a sparse sweep can simply sit between two unmeasured conventional points.

## Drainage diagnostic

Both surfaces go through the same routing, with the same parameters:

1. priority-flood depression filling with a 1e-6 m gradient across flats,
2. D8 steepest descent by ground slope,
3. contributing-area accumulation,
4. stream cells where the contributing area reaches 500 cells (0.05 km2 at 10 m),
5. basin membership by pointer jumping.

The headline number is the Jaccard index of the two stream masks,
J = |S ∩ S'| / |S ∪ S'|, with J = 0 when both are empty. Exact-cell Jaccard punishes a one-cell
shift hard, so receiver agreement, recall and basin counts are reported next to it. This is a
self-consistency check between two surfaces. It says nothing about real discharge, culverts or
sewers.

## Sparse corrections

The encoder knows the reference stream network. It stores the coarse quantised field plus exact
corrections (to the 1 cm quantum) on a band of cells around the reference streams. Positions are
stored as varint gaps between sorted indexes and values as zigzag varints, then compressed with
zstd or gzip, whichever is smaller. All of it is charged.
The decoder needs nothing it does not receive. Off the band the error bound is the coarse target;
on it, the fine quantum.

## Neural families

All networks map a lattice coordinate (and for some families a tile index) to a height. Families:

| Family | Idea |
|---|---|
| mlp | Plain MLP on raw coordinates; the control |
| siren | Sine activations (Sitzmann et al. 2020) |
| fourier | Random Fourier features before an MLP |
| bandlimited | Multiplicative filter network with bounded frequencies per layer (BACON-like) |
| grid | Multiresolution feature grid (dense or hashed) plus a small decoder |
| codegrid | Per-tile latent codes on a grid, shared decoder |
| shared | One decoder shared by all tiles, a latent code per tile |
| residual / hybrid | A stored conventional coarse grid plus a network for the residual |
| liif | Local implicit image function: a latent grid decoded by a small network at each query |

Architectures and recipes were chosen by a per-family Optuna search under a byte ceiling, then the
finalists were retrained across seeds, serialised and evaluated from the stored weights. The
quantised ladder stores each finalist at float16 and at 8, 6 and 4 bit, both after training (PTQ)
and with quantisation-aware fine-tuning (QAT), and measures error after quantisation.

## Landscape physics

The teacher is an idealised landscape-evolution model on a node grid,

  dh/dt = U + div(D grad h) - K A^m |grad h|^n,

with uplift U (m/yr), hillslope diffusivity D (m2/yr), contributing area A and stream-power
parameters K, m, n. The audit checks timestep and grid refinement, the closed-domain balance and
the slope-area law S = (U/K)^(1/n) A^(-m/n). Uplifting a closed domain including its outlet
produces a rising flat surface, so boundary conditions are part of the model.

### Learned conservative fluxes

For the hillslope term, write the update in flux form over cell faces:

  h_i(t+dt) = h_i(t) + dt s_i - (dt / a_i) sum_j F_ij,   with F_ij = -F_ji.

Summed over the domain with cell areas, every interior face cancels:

  sum_i a_i (h_i(t+dt) - h_i(t)) = dt sum_i a_i s_i - dt (flux through the boundary).

A network that predicts one flux per face and applies it with opposite signs to the two cells
conserves the integral for any weights. This guarantees the balance, not accuracy, stability or
realistic sediment transport.

The target is nonlinear critical-slope diffusion (Roering type, critical slope 0.6), which linear
diffusion cannot represent at any D. Arms, all with the same data and training budget and about the
same capacity (28,641 and 28,353 parameters):

* `flux`: one learned flux per face, conservative by construction,
* `kfield`: a learned per-cell diffusivity inside the five-point Laplacian, not conservative,
* `penalty-w`: the same K-field with a conservation penalty of weight w in the loss,
* linear diffusion with the teacher's own coefficient as a reference.

Every arm is trained on five seeds and the penalty weight is swept from 1e-4 to 10, so the
comparison is against a tuned soft constraint, not one arbitrary weight. Each trained arm predicts
the rate of height change on 24 unseen surfaces. The error is the mean absolute difference from the
teacher's rate (m/yr), and the conservation residual is the domain total of the predicted rate over
the total of its absolute value. A 64-step rollout of 200 years per step then records the drift of
the integrated height and whether the surface stays finite.

### Browser kernel

The lab in the browser runs the same operators in Rust compiled to WebAssembly: linear and
nonlinear diffusion in flux form, the three learned arms with exported weights, and closed,
fixed or periodic boundaries. Parity tests compare it against fixtures exported from the Python
implementation.
