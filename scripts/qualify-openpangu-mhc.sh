#!/usr/bin/env bash
# Run in the qualified NPU Docker image; Rust exporter is cross-built locally.
set -eo pipefail
if (( $# != 2 )); then echo 'usage: qualify-openpangu-mhc.sh CHECKPOINT OUTPUT_DIR' >&2; exit 2; fi
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/ascend-toolkit/vendors/omni_training_custom_transformer/bin/set_env.bash
mkdir -p "$2/fixtures" "$2/reports"
for pair in phi:phi.weight alpha:branch_alpha beta:branch_beta gamma:norm_gamma; do
  .deploy/export_openpangu_tensor "$1" "model.layers.0.attn_mhc_module.${pair#*:}" "$2/fixtures/${pair%%:*}.bin" --allow-stale-index-size
done
cmake -S native -B native-build -DCMAKE_BUILD_TYPE=Release
cmake --build native-build --target openpangu_mhc_probe -j4
for batch in 1 4; do
  timeout 120 native-build/openpangu_mhc_probe "$2/fixtures" "$batch" > "$2/reports/batch$batch.json"
  cat "$2/reports/batch$batch.json"
done
sha256sum native-build/openpangu_mhc_probe /usr/local/Ascend/ascend-toolkit/vendors/omni_training_custom_transformer/op_api/lib/libcust_opapi.so > "$2/reports/artifacts.sha256"
ldd native-build/openpangu_mhc_probe > "$2/reports/dependencies.txt"
