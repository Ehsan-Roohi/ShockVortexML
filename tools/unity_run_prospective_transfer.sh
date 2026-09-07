#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
INPUT_ROOT=${INPUT_ROOT:?set INPUT_ROOT to derived_selected_20260907_v3}
OUTPUT_ROOT=${OUTPUT_ROOT:?set OUTPUT_ROOT to a new output directory}
ENV_ROOT=${ENV_ROOT:-${OUTPUT_ROOT}_venv}

if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
  python -m venv "$ENV_ROOT"
  "$ENV_ROOT/bin/pip" install --upgrade pip
  "$ENV_ROOT/bin/pip" install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
  "$ENV_ROOT/bin/pip" install numpy scipy
fi

mkdir -p "$OUTPUT_ROOT/references" "$OUTPUT_ROOT/predictions"
printf '%s\n' '{"rho_inf":1.0,"pressure_inf":0.7142857142857143,"u_inf_x":2.2,"u_inf_y":0.0,"gamma":1.4,"reference_length":1.0,"source":"registered_case_metadata"}' > "$OUTPUT_ROOT/references/m2p2.json"
printf '%s\n' '{"rho_inf":1.0,"pressure_inf":0.7142857142857143,"u_inf_x":2.7,"u_inf_y":0.0,"gamma":1.4,"reference_length":1.0,"source":"registered_case_metadata"}' > "$OUTPUT_ROOT/references/m2p7.json"

cd "$ROOT"
export PYTHONPATH="$ROOT/src"
"$ENV_ROOT/bin/python" tools/prospective_transfer_20260906.py verify

count=0
while IFS= read -r input; do
  case_id=$(basename "$(dirname "$input")")
  stem=$(basename "$input" .npz)
  reference="$OUTPUT_ROOT/references/m2p2.json"
  [[ "$case_id" == "ellipse_m2p7" ]] && reference="$OUTPUT_ROOT/references/m2p7.json"
  mkdir -p "$OUTPUT_ROOT/predictions/$case_id"
  "$ENV_ROOT/bin/python" tools/prospective_transfer_20260906.py infer \
    --input "$input" \
    --reference "$reference" \
    --output "$OUTPUT_ROOT/predictions/$case_id/$stem.npz"
  count=$((count + 1))
done < <(find "$INPUT_ROOT" -mindepth 2 -maxdepth 2 -type f -name '*.npz' | sort)

[[ "$count" -eq 9 ]]
sha256sum "$OUTPUT_ROOT"/predictions/*/*.npz > "$OUTPUT_ROOT/PREDICTIONS.sha256"
printf 'status=PASS\ncommit=%s\ninputs=%d\nbranch=ML_ONLY\n' "$(git rev-parse HEAD)" "$count" > "$OUTPUT_ROOT/RUN_OK.txt"
cat "$OUTPUT_ROOT/RUN_OK.txt"
