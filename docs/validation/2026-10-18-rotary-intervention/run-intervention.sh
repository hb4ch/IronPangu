set -eo pipefail
cd /data/p00603624/inferfabric/openpangu92
export INFERFABRIC_REFERENCE_ROTARY_OVERRIDE=$PWD/diagnostic-native-rope
export REFERENCE_NPU_DEVICES=0,1,2,3,4,5,6,7 HCCL_IF_BASE_PORT=47000
bash scripts/reference-openpangu-decoder.sh --cases extended-answer64-reference.json --output extended-answer64-rope-intervention2.json --max-model-len 65536 --max-num-batched-tokens 4096 --kv-cache-dtype int8_ds_mla --respect-eos > extended-answer64-rope-intervention2.log 2>&1
