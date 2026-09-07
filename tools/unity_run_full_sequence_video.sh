#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE_RUN=${SOURCE_RUN:?set SOURCE_RUN to the registered three-case run}
WORK_ROOT=${WORK_ROOT:?set a new derived output root}
ENV_ROOT=${ENV_ROOT:?set the frozen inference environment}
mkdir -p "$WORK_ROOT"
cd "$ROOT"; export PYTHONPATH="$ROOT/src"
"$ENV_ROOT/bin/python" tools/prospective_reduce_full_sequence.py --run "$SOURCE_RUN" --output "$WORK_ROOT/inputs"
INPUT_ROOT="$WORK_ROOT/inputs" OUTPUT_ROOT="$WORK_ROOT/ml_only" ENV_ROOT="$ENV_ROOT" EXPECTED_INPUTS=243 bash tools/unity_run_prospective_transfer.sh
"$ENV_ROOT/bin/pip" install -q matplotlib h5py
"$ENV_ROOT/bin/python" tools/prospective_two_panel_video.py --inputs "$WORK_ROOT/inputs" --predictions "$WORK_ROOT/ml_only/predictions" --output "$WORK_ROOT/two_panel"
printf 'status=PASS\nframes=243\nlayout=physics_left_ml_right\ncommit=%s\n' "$(git rev-parse HEAD)" > "$WORK_ROOT/RUN_OK_FULL_SEQUENCE_VIDEO.txt"
cat "$WORK_ROOT/RUN_OK_FULL_SEQUENCE_VIDEO.txt"
