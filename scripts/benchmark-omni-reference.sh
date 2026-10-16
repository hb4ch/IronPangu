#!/usr/bin/env bash
# Offline comparison only; OmniInfer/Torch are never part of native serving.
set -eo pipefail
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/ascend-toolkit/vendors/omni_training_custom_transformer/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/vendors/omni_custom_transformer/bin/set_env.bash
export OMNI_VLLM_PATCHES_DIR=high_throughout
export VLLM_PLUGINS=omni-npu,omni_npu_patches,omni_custom_models
export VLLM_DISABLE_PYNCCL=1
export ASCEND_RT_VISIBLE_DEVICES="${REFERENCE_NPU_DEVICES:-0,1,2,3,4,5,6,7}"
export OMNI_CUSTOM_MODEL_CONFIG_PATH="$(dirname "$(realpath "$0")")/benchmark-omni-reference-config.json"
export HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120
python3 "$(dirname "$0")/benchmark-omni-reference.py" "$@"
