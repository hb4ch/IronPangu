set -eo pipefail
cd /data/p00603624/inferfabric/openpangu92
python3 scripts/make-long-context-cases.py --tokenizer /models/92B/tokenizer.json --output extended-answer-input.json --contexts 8192 65536 --positions middle --output-tokens 128 --task extended-answer
export REFERENCE_NPU_DEVICES=0,1,2,3,4,5,6,7 HCCL_IF_BASE_PORT=46000
bash scripts/reference-openpangu-decoder.sh --cases extended-answer-input.json --output extended-answer-reference.json --max-model-len 65536 --max-num-batched-tokens 4096 --kv-cache-dtype int8_ds_mla --respect-eos > extended-answer-reference.log 2>&1
