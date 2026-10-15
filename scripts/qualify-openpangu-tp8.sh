#!/usr/bin/env bash
# Run inside the assigned NPU Docker container, from a staged repository root.
# Requires locally cross-compiled inferfabric-npu and export_openpangu_linear.
set -eo pipefail
if (( $# != 2 )); then echo 'usage: qualify-openpangu-tp8.sh CHECKPOINT OUTPUT_DIR' >&2; exit 2; fi
checkpoint=$1
output=$2
source /usr/local/Ascend/cann/set_env.sh
export HCCL_OP_EXPANSION_MODE=HOST HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120
mkdir -p "$output/fixtures" "$output/reports"
./inferfabric-npu plan-openpangu examples/openpangu/92b-tp8-ep8.inferfabric "$checkpoint" "$output/plan.json" --allow-stale-index-size
./export_openpangu_linear "$checkpoint" model.layers.0.mlp.down_proj "$output/fixtures/tp.bin" --allow-stale-index-size
for rank in 0 1 2 3 4 5 6 7; do
  expert=$((rank*32))
  for projection in gate up down; do
    ./export_openpangu_linear "$checkpoint" "model.layers.2.mlp.experts.$expert.${projection}_proj" "$output/fixtures/rank$rank-$projection.bin" --allow-stale-index-size
  done
done
cmake -S native -B native-build -DCMAKE_BUILD_TYPE=Release
cmake --build native-build --target openpangu_parallel_probe -j4
for batch in 1 4; do
  timeout 240 ./native-build/openpangu_parallel_probe "$output/fixtures" "$batch" > "$output/reports/batch$batch.json"
  cat "$output/reports/batch$batch.json"
done
