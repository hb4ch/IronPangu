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
test -f "$2/execution.bin"
mkdir -p "$3/plan"
cp "$2/"*.bin "$3/plan/"
if [[ -f "$2/arena.json" ]]; then
  cp "$2/arena.json" "$2/"arena-rank-*.txt "$3/plan/"
fi
cp "$2/execution.json" "$3/execution.json"
cp native-build/openpangu_decode "$3/openpangu_decode"
selected_swiglu=${INFERFABRIC_SWIGLU_LIBRARY:-native-build/libinferfabric_swiglu.so}
cp "$selected_swiglu" "$3/selected-swiglu.so"
export INFERFABRIC_SWIGLU_LIBRARY=$(realpath "$3/selected-swiglu.so")
python3 scripts/qualify-openpangu-decoder.py --checkpoint "$1" --plan "$3/plan" --binary "$3/openpangu_decode" --output "$3"
sha256sum "$3/openpangu_decode" "$3/selected-swiglu.so" "$1/config.json" "$1/tokenizer.json" "$3/plan/"*.bin > "$3/artifacts.sha256"
if [[ -f "$3/plan/arena.json" ]]; then
  sha256sum "$3/plan/arena.json" "$3/plan/"arena-rank-*.txt >> "$3/artifacts.sha256"
fi
ldd "$3/openpangu_decode" > "$3/dependencies.txt"
