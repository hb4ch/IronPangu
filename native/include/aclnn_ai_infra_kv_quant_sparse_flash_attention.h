/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_AI_INFRA_KV_QUANT_SPARSE_FLASH_ATTENTION_OP_H_
#define OP_API_INC_LEVEL2_AI_INFRA_KV_QUANT_SPARSE_FLASH_ATTENTION_OP_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnAiInfraKvQuantSparseFlashAttention 的第一段接口，根据具体的计算流程，计算workspace大小。
 * @param query 查询张量（必需）
 * @param key 键张量（必需）

 * @return ACLNN_SUCCESS 成功
 */
aclnnStatus aclnnAiInfraKvQuantSparseFlashAttentionGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *sparseIndices,
    const aclTensor *keyDequantScaleOptional,
    const aclTensor *valueDequantScaleOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    const aclTensor *keySinkOptional,
    const aclTensor *valueSinkOptional,
    double scaleValue,
    int64_t keyQuantMode,
    int64_t valueQuantMode,
    int64_t sparseBlockSize,
    char *layoutQueryOptional,
    char *layoutKvOptional,
    int64_t sparseMode,
    int64_t preTokens,
    int64_t nextTokens,
    int64_t attentionMode,
    int64_t quantScaleRepoMode,
    int64_t tileSize,
    int64_t ropeHeadDim,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnAiInfraKvQuantSparseFlashAttention的第二段接口，用于执行计算。
 */
aclnnStatus aclnnAiInfraKvQuantSparseFlashAttention(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

__attribute__((visibility("default"))) aclnnStatus aclnnAiInfraKvQuantSparseFlashAttentionGetMaxWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *sparseIndices,
    const aclTensor *keyDequantScaleOptional,
    const aclTensor *valueDequantScaleOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    const aclTensor *keySinkOptional,
    const aclTensor *valueSinkOptional,
    double scaleValue,
    int64_t keyQuantMode,
    int64_t valueQuantMode,
    int64_t sparseBlockSize,
    char *layoutQueryOptional,
    char *layoutKvOptional,
    int64_t sparseMode,
    int64_t preTokens,
    int64_t nextTokens,
    int64_t attentionMode,
    int64_t quantScaleRepoMode,
    int64_t tileSize,
    int64_t ropeHeadDim,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_SPARSE_FLASH_ATTENTION_PIONEER_H_