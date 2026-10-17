set -eo pipefail
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/ascend-toolkit/vendors/omni_training_custom_transformer/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/vendors/omni_custom_transformer/bin/set_env.bash
export HCCL_OP_EXPANSION_MODE=HOST HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120 HCCL_IF_BASE_PORT=46200
export INFERFABRIC_DECODE_GRAPH=1 INFERFABRIC_EXPERIMENTAL_LONG_CONTEXT=1
cd /data/p00603624/inferfabric/openpangu92
python3 scripts/prepare-sequence-fixture.py --plan bulk-domains8k-frozen-plan --references extended-answer-reference.json --peers batched16-preparation-reference.json --peer-case arithmetic --output extended-answer-fixture
python3 scripts/qualify-sequence-serving.py --frozen-only --batch-trace --binary native-build-bulk-domains/program_engine_probe --plan bulk-domains8k-frozen-plan --checkpoint /models/92B --requests extended-answer-fixture/requests.txt --bindings extended-answer-fixture/page-bindings.txt --reference-output extended-answer-fixture/reference.jsonl --output-prefix extended-answer-native
