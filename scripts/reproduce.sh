#!/usr/bin/env bash
# Regenerate every published report from the downloaded data, then the tables, figures and
# browser bundle. Needs `uv sync --all-extras`, a CUDA GPU for the neural and physics steps,
# and about two hours on one workstation. Reports go to $GEONEURAL_HOME/results/reproduced.
set -euo pipefail
cd "$(dirname "$0")/.."
export GEONEURAL_HOME="${GEONEURAL_HOME:-$PWD/.data}"
R="$GEONEURAL_HOME/results/reproduced"
A="$GEONEURAL_HOME/atlases/essen-ruhr/atlas.json"
G="$GEONEURAL_HOME/geology/essen-ruhr"
mkdir -p "$R" "$GEONEURAL_HOME/geology"
run() { echo "== geoneural $*"; uv run geoneural "$@"; }

# Data: the shipped Essen tiles and geology, plus the provider files for the registration checks.
[ -f "$A" ] || run prepare --input data/sample/essen-ruhr/raw/input.json --out "$(dirname "$A")"
[ -d "$G" ] || cp -r data/sample/essen-ruhr/geology "$G"
# Height benchmarks (hfp_pl.csv, hfp_plzf.csv) and the DGM1 tile metadata (dgm1_meta.zip,
# dgm1_tiff_index.xml) come from opengeodata.nrw.de; see docs/data.md.

# Conventional codecs, drainage and corrections (CPU).
run tournament --atlas "$A" --out "$R/tournament-essen.json"
run account --package "$(dirname "$A")" --out "$R/accounting-essen.json"
run drainage --atlas "$A" --out "$R/drainage-essen.json"
run corrections --atlas "$A" --out "$R/corrections-essen.json"
run envelope --atlas "$A" --out "$R/dense-envelope-essen.json"
run certify-seams --atlas "$A" --out "$R/seams-essen.json"
run verify-lattice --atlas "$A" --out "$R/lattice-essen.json"
run landmarks-dlm --atlas "$A" --out "$R/dlm-essen.json"
[ -d "$GEONEURAL_HOME/landmarks/nrw-hfp" ] && run landmarks-hfp --atlas "$A" --out "$R/height-benchmarks-essen.json"
[ -d "$GEONEURAL_HOME/epochs/nrw-dgm1" ] && run epochs --atlas "$A" --out "$R/epochs-essen-ruhr.json"

# Neural representations (GPU). The finalist configurations come from the architecture search.
run codec-fit --atlas "$A" --chosen results/chosen/codec-fit-finalists.json --out "$R/neural-codec-fit.json" \
  --save-fields "$GEONEURAL_HOME/fields"
run evaluate-checkpoint --atlas "$A" --model results/models/siren-codec-fit/model.json --out "$R/siren-checkpoint.json" \
  --save-field "$GEONEURAL_HOME/fields/checkpoint--siren-codec-fit.npy"
run quantised-ladder --atlas "$A" --chosen results/chosen/quantised-ladder-finalists.json --out "$R/quantised-ladder.json"
run decode-latency --atlas "$A" --chosen results/chosen/quantised-ladder-finalists.json --out "$R/decode-latency.json"
for mode in film concat; do
  run context-ablation --atlas "$A" --geology "$G" --conditioning "$mode" --out "$R/geology-$mode.json"
  run context-ablation --atlas "$A" --geology "$G" --conditioning "$mode" --base-level 2 --base-target-m 1.0 \
    --out "$R/geology-residual-$mode.json"
done

# Landscape physics.
run teacher-audit --out "$R/teacher-audit.json"
run flux-closure --out "$R/flux-closure.json"
run flux-closure-seeds --out "$R/flux-closure-seeds.json"
run event-field --out "$R/event-field.json"
run ensemble --out "$GEONEURAL_HOME/ensemble/main"
run emulator --ensemble "$GEONEURAL_HOME/ensemble/main" --out "$R/emulator.json"

# Super-resolution needs the 1 m data of three regions (about 800 MB of downloads):
#   geoneural fetch --preset <region> --source-spacing 1 --tile-m 1024 --max-mib 2048 --out .data/raw/<region>-1m
#   geoneural fine-reference --preset <region>
if [ -f "$GEONEURAL_HOME/fine/rothaar-sauerland/coarse_from_operator.npy" ]; then
  run distillation --out "$R/distillation.json"
  run superres-neural --out "$R/superres-neural.json"
  run superres-classical --out "$R/superres-classical.json"
fi

# Tables, figures and the browser bundle.
run publish
run candidates
run summary
run figures
run export-web
