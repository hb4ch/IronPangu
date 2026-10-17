set -eo pipefail
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/ascend-toolkit/vendors/omni_training_custom_transformer/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/vendors/omni_custom_transformer/bin/set_env.bash
export HCCL_OP_EXPANSION_MODE=HOST HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120 HCCL_IF_BASE_PORT=47400
export INFERFABRIC_DECODE_GRAPH=1 INFERFABRIC_EXPERIMENTAL_LONG_CONTEXT=1
cd /data/p00603624/inferfabric/openpangu92
python3 scripts/prepare-sequence-fixture.py --plan bf16-collectives-source --references device-zero-apc8k-fixture/cases.json --output bf16-collectives8k-fixture
python3 scripts/qualify-sequence-serving.py --batch-trace --binary native-build-bulk-domains/program_engine_probe --plan bf16-collectives-source --checkpoint /models/92B --requests bf16-collectives8k-fixture/requests.txt --bindings bf16-collectives8k-fixture/page-bindings.txt --reference-output bf16-collectives8k-fixture/reference.jsonl --output-prefix bf16-collectives8k-bound
