#!/usr/bin/env bash
set -eo pipefail
if (( $# != 1 )); then echo 'usage: qualify-openpangu-attention.sh OUTPUT_DIR' >&2; exit 2; fi
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/ascend-toolkit/vendors/omni_custom_transformer/bin/set_env.bash
mkdir -p "$1"
cmake -S native -B native-build -DCMAKE_BUILD_TYPE=Release
cmake --build native-build --target openpangu_attention_probe -j4
for heads in 6 48; do
 timeout 120 native-build/openpangu_attention_probe "$heads" > "$1/heads-$heads.json"
 cat "$1/heads-$heads.json"
done
sha256sum native-build/openpangu_attention_probe /usr/local/Ascend/ascend-toolkit/vendors/omni_custom_transformer/op_api/lib/libcust_opapi.so > "$1/artifacts.sha256"
ldd native-build/openpangu_attention_probe > "$1/dependencies.txt"
