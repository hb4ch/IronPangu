#!/usr/bin/env bash
# Offline diagnostic qualification; never a serving frontend.
set -eo pipefail
if (( $# != 3 )); then echo 'usage: qualify-openpangu-decoder.sh CHECKPOINT LOAD_PLAN_DIR OUTPUT_DIR' >&2; exit 2; fi
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/ascend-toolkit/vendors/omni_training_custom_transformer/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/vendors/omni_custom_transformer/bin/set_env.bash
export HCCL_OP_EXPANSION_MODE=HOST HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120
cmake -S native -B native-build -DCMAKE_BUILD_TYPE=Release
cmake --build native-build --target openpangu_decode -j4
mkdir -p "$3"
cp native-build/openpangu_decode "$3/openpangu_decode"
python3 scripts/qualify-openpangu-decoder.py --checkpoint "$1" --plan "$2" --binary "$3/openpangu_decode" --output "$3"
sha256sum "$3/openpangu_decode" "$1/config.json" "$1/tokenizer.json" > "$3/artifacts.sha256"
ldd "$3/openpangu_decode" > "$3/dependencies.txt"
