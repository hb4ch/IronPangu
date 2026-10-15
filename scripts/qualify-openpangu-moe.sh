#!/usr/bin/env bash
# All commands run inside the NPU Docker container. Cross-build Rust locally first.
set -eo pipefail
if (( $# != 2 )); then echo 'usage: qualify-openpangu-moe.sh CHECKPOINT OUTPUT_DIR' >&2; exit 2; fi
source /usr/local/Ascend/cann/set_env.sh
export HCCL_OP_EXPANSION_MODE=HOST HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120
mkdir -p "$2/fixtures" "$2/reports"
./.deploy/export_openpangu_moe "$1" 2 "$2/fixtures" "$2/fixtures.json" --allow-stale-index-size
cmake -S native -B native-build -DCMAKE_BUILD_TYPE=Release
cmake --build native-build --target openpangu_moe_probe openpangu_parallel_probe openpangu_linear_probe -j4
for batch in 1 4; do
  timeout 480 native-build/openpangu_moe_probe "$2/fixtures" "$batch" > "$2/reports/batch$batch.json"
  cat "$2/reports/batch$batch.json"
done
sha256sum native-build/openpangu_moe_probe .deploy/export_openpangu_moe > "$2/reports/artifacts.sha256"
