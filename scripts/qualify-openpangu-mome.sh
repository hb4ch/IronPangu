#!/usr/bin/env bash
set -eo pipefail
if (( $# != 2 )); then echo 'usage: qualify-openpangu-mome.sh CHECKPOINT OUTPUT_DIR' >&2; exit 2; fi
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/ascend-toolkit/vendors/omni_custom_transformer/bin/set_env.bash
mkdir -p "$2/fixtures" "$2/reports"
for name in qa compresskv o; do
 .deploy/export_openpangu_tensor "$1" "model.layers.0.self_attn.${name}_conv.weight" "$2/fixtures/$name.bin" --allow-stale-index-size
done
cmake -S native -B native-build -DCMAKE_BUILD_TYPE=Release
cmake --build native-build --target openpangu_mome_probe -j4
for name in qa compresskv o; do
 timeout 120 native-build/openpangu_mome_probe "$2/fixtures/$name.bin" > "$2/reports/$name.json"
 cat "$2/reports/$name.json"
done
sha256sum native-build/openpangu_mome_probe /usr/local/Ascend/ascend-toolkit/vendors/omni_custom_transformer/op_api/lib/libcust_opapi.so > "$2/reports/artifacts.sha256"
ldd native-build/openpangu_mome_probe > "$2/reports/dependencies.txt"
