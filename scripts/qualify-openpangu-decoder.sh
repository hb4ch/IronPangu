#!/usr/bin/env bash
# Offline diagnostic qualification; never a serving frontend.
set -eo pipefail
if (( $# != 3 )); then echo 'usage: qualify-openpangu-decoder.sh CHECKPOINT LOAD_PLAN_DIR OUTPUT_DIR' >&2; exit 2; fi
source /usr/local/Ascend/cann/set_env.sh
export HCCL_OP_EXPANSION_MODE=HOST HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120
cmake -S native -B native-build -DCMAKE_BUILD_TYPE=Release
cmake --build native-build --target openpangu_decode -j4
python3 scripts/qualify-openpangu-decoder.py --checkpoint "$1" --plan "$2" --output "$3"
sha256sum native-build/openpangu_decode "$1/config.json" "$1/tokenizer.json" > "$3/artifacts.sha256"
ldd native-build/openpangu_decode > "$3/dependencies.txt"
