#!/usr/bin/env bash
set -euo pipefail
: "${WORK_ROOT:?}" "${RUNTIME_ROOT:?}" "${ENV_ROOT:?}"
OUT="$WORK_ROOT/rear_shock_hysteresis_20260907_v2"
[[ ! -e "$OUT" ]] || { echo "ERROR: output exists: $OUT" >&2; exit 2; }
mkdir -p "$OUT"
cd "$RUNTIME_ROOT"
"$ENV_ROOT/bin/python" tools/rear_shock_hysteresis_20260907.py \
  --inputs "$WORK_ROOT/inputs" \
  --predictions "$WORK_ROOT/rear_shock_repair_20260907_v1/predictions" \
  --output "$OUT/predictions"
"$ENV_ROOT/bin/python" tools/prospective_two_panel_video.py \
  --inputs "$WORK_ROOT/inputs" --predictions "$OUT/predictions" --output "$OUT/two_panel"
printf 'status=PASS\nmethod=learned-probability hysteresis plus entropy-free density-gradient support\n' > "$OUT/RUN_OK.txt"
