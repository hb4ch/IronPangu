#!/usr/bin/env bash
set -eo pipefail
if (( $# != 3 )); then echo 'usage: qualify-openpangu-load.sh CHECKPOINT DSL OUTPUT_DIR' >&2; exit 2; fi
source /usr/local/Ascend/cann/set_env.sh
mkdir -p "$3/reports"
.deploy/export_openpangu_load_plan "$1" "$2" "$3/plan" --allow-stale-index-size
cmake -S native -B native-build -DCMAKE_BUILD_TYPE=Release
cmake --build native-build --target openpangu_load_probe -j4
python3 native/probes/test_openpangu_load_plan.py native-build/openpangu_load_probe > "$3/reports/manifest-tests.txt"
pids=()
for rank in {0..7}; do
 timeout 600 native-build/openpangu_load_probe "$3/plan/rank-$rank.bin" "$1" > "$3/reports/rank-$rank.json" 2> "$3/reports/rank-$rank.stderr" &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then cat "$3"/reports/*.stderr; exit 1; fi
cat "$3"/reports/rank-*.json
sha256sum native-build/openpangu_load_probe .deploy/export_openpangu_load_plan "$3"/plan/rank-*.bin > "$3/reports/artifacts.sha256"
ldd native-build/openpangu_load_probe > "$3/reports/dependencies.txt"
