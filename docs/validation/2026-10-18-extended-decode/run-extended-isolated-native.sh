set -eo pipefail
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/ascend-toolkit/vendors/omni_training_custom_transformer/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/vendors/omni_custom_transformer/bin/set_env.bash
export HCCL_OP_EXPANSION_MODE=HOST HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120 HCCL_IF_BASE_PORT=46600
export INFERFABRIC_DECODE_GRAPH=1 INFERFABRIC_EXPERIMENTAL_LONG_CONTEXT=1
cd /data/p00603624/inferfabric/openpangu92
python3 - <<'PY'
import json
from pathlib import Path
cases=json.loads(Path('extended-answer-reference.json').read_text())
with Path('extended-answer64-reference.json').open('x') as f:json.dump(cases[-1:],f)
PY
python3 scripts/prepare-sequence-fixture.py --plan bulk-domains8k-frozen-plan --references extended-answer64-reference.json --output extended-answer64-isolated-fixture
python3 scripts/qualify-sequence-serving.py --frozen-only --batch-trace --binary native-build-bulk-domains/program_engine_probe --plan bulk-domains8k-frozen-plan --checkpoint /models/92B --requests extended-answer64-isolated-fixture/requests.txt --bindings extended-answer64-isolated-fixture/page-bindings.txt --reference-output extended-answer64-isolated-fixture/reference.jsonl --output-prefix extended-answer64-isolated-native
