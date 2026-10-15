#!/usr/bin/env bash
# Inside the NPU container; exporter must have been cross-compiled locally.
set -eo pipefail
if (( $# != 2 )); then echo 'usage: qualify-openpangu-router.sh CHECKPOINT OUTPUT_DIR' >&2; exit 2; fi
source /usr/local/Ascend/cann/set_env.sh
mkdir -p "$2/fixtures" "$2/reports"
./.deploy/export_openpangu_tensor "$1" model.layers.2.mlp.gate.weight "$2/fixtures/router-weight.bin" --allow-stale-index-size
./.deploy/export_openpangu_tensor "$1" model.layers.2.mlp.e_score_correction_bias "$2/fixtures/router-bias.bin" --allow-stale-index-size
cmake -S native -B native-build -DCMAKE_BUILD_TYPE=Release
cmake --build native-build --target openpangu_router_probe -j4
for batch in 1 4; do
  timeout 120 native-build/openpangu_router_probe "$2/fixtures" "$batch" > "$2/reports/batch$batch.json"
  cat "$2/reports/batch$batch.json"
done
