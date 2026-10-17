/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_IF_OWNED_ACLNN_AI_INFRA_KV_RMS_NORM_ROPE_CACHE_H_
#define OP_API_IF_OWNED_ACLNN_AI_INFRA_KV_RMS_NORM_ROPE_CACHE_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief ifOwnedAiInfraKvRmsNormRopeCache 的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 */
aclnnStatus ifOwnedAiInfraKvRmsNormRopeCacheGetWorkspaceSize(
    const aclTensor* kv, const aclTensor* gamma, const aclTensor* cos, const aclTensor* sin, const aclTensor* index,
    aclTensor* kCacheRef, aclTensor* ckvCacheRef, const aclTensor* kRopeScaleOptional,
    const aclTensor* ckvScaleOptional, const aclTensor* kRopeOffsetOptional, const aclTensor* cKvOffsetOptional,
    const aclTensor* vOptional, double epsilon, char* cacheModeOptional, char* rotaryModeOptional,
    char* quantModeOptional, bool isOutputKv, aclTensor* kRopeOut, aclTensor* cKvOut, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief ifOwnedAiInfraKvRmsNormRopeCache 的第二段接口，用于执行计算。
 */
aclnnStatus ifOwnedAiInfraKvRmsNormRopeCache(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_IF_OWNED_ACLNN_AI_INFRA_KV_RMS_NORM_ROPE_CACHE_H_