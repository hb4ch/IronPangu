#!/usr/bin/env bash
# OFFLINE GOLDEN REFERENCE ONLY: Torch/OmniInfer are excluded from native serving.
set -eo pipefail
source /usr/local/Ascend/cann/set_env.sh
export OMNI_VLLM_PATCHES_DIR=high_throughout
export VLLM_PLUGINS=omni-npu,omni_npu_patches,omni_custom_models
export VLLM_DISABLE_PYNCCL=1
export ASCEND_RT_VISIBLE_DEVICES="${REFERENCE_NPU_DEVICES:-8,9,10,11,12,13,14,15}"
export OMNI_CUSTOM_MODEL_CONFIG_PATH="$PWD/scripts/reference-openpangu-decoder-config.json"
export HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120
python3 scripts/reference-openpangu-decoder.py "$@"
