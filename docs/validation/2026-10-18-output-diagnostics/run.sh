set -eo pipefail
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/ascend-toolkit/vendors/omni_training_custom_transformer/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/vendors/omni_custom_transformer/bin/set_env.bash
export HCCL_OP_EXPANSION_MODE=HOST HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120 HCCL_IF_BASE_PORT=48400
export INFERFABRIC_PREPARE_ARENA=1
export INFERFABRIC_DECODE_GRAPH=1 INFERFABRIC_EXPERIMENTAL_LONG_CONTEXT=1 INFERFABRIC_PACKED_PREFILL=1 INFERFABRIC_OUTPUT_DIAGNOSTICS=1
export INFERFABRIC_SEQUENCE_BATCH=1 INFERFABRIC_INTERLEAVE_RESIDENTS=1
cd /data/p00603624/inferfabric/openpangu92
mkdir bf16-collectives64-batch-topk-discovery
python3 - <<'PY'
import json
from pathlib import Path
c=json.loads(Path('extended-answer64-reference.json').read_text())[0]
Path('bf16-collectives64-batch-topk-discovery/requests.txt').write_text('8 '+' '.join(map(str,c['input_ids']))+'\n')
PY
export INFERFABRIC_PAGE_BINDINGS=$PWD/bf16-collectives-extended-fixture/page-bindings.txt
export INFERFABRIC_PREPARED_AUDIT_DIR=$PWD/bf16-collectives64-batch-topk-discovery/prepared
export INFERFABRIC_TRANSFER_AUDIT_DIR=$PWD/bf16-collectives64-batch-topk-discovery/transfers
native-build-batch-diagnostics/program_engine_probe bf16-collectives-source /models/92B bf16-collectives64-batch-topk-discovery/requests.txt > bf16-collectives64-batch-topk-discovery/tokens.jsonl 2> bf16-collectives64-batch-topk-discovery/stderr.log
