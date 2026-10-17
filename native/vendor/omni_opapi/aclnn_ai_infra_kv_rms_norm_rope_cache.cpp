/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_ai_infra_kv_rms_norm_rope_cache.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "ai_infra_kv_rms_norm_rope_cache.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
static constexpr size_t KV_DIM_NUM = 4;
static constexpr size_t K_CACHE_DIM_NUM = 4;
static constexpr size_t CKV_CACHE_DIM_NUM = 4;

static const std::initializer_list<DataType> KV_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16, DataType::DT_BF16};
static const std::initializer_list<DataType> CACHE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_INT8, DataType::DT_FLOAT8_E4M3FN, DataType::DT_HIFLOAT8};
static const std::initializer_list<DataType> INDEX_DTYPE_SUPPORT_LIST = {DataType::DT_INT64};

static aclnnStatus CheckAiInfraKvRmsNormRopeCacheParams(
    const aclTensor* kv, const aclTensor* gamma, const aclTensor* cos, const aclTensor* sin, const aclTensor* index,
    aclTensor* kCacheRef, aclTensor* ckvCacheRef, char* cacheMode, char* rotaryMode, char* quantMode,
    const aclTensor* kRopeOut, const aclTensor* cKvOut)
{
    OP_CHECK_NULL(kv, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(gamma, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(cos, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(sin, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(index, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(ckvCacheRef, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(kRopeOut, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(cKvOut, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(cacheMode, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(rotaryMode, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(quantMode, return ACLNN_ERR_INNER_NULLPTR);

    OP_CHECK_WRONG_DIMENSION(kv, KV_DIM_NUM, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(ckvCacheRef, CKV_CACHE_DIM_NUM, return ACLNN_ERR_PARAM_INVALID);

    OP_CHECK_DTYPE_NOT_SUPPORT(kv, KV_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(gamma, KV_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(cos, KV_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(sin, KV_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(index, INDEX_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(ckvCacheRef, CACHE_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    if (kCacheRef != nullptr) {
        OP_CHECK_NULL(kCacheRef, return ACLNN_ERR_INNER_NULLPTR);
        OP_CHECK_WRONG_DIMENSION(kCacheRef, K_CACHE_DIM_NUM, return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_DTYPE_NOT_SUPPORT(kCacheRef, CACHE_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus AiInfraKvRmsNormRopeCacheCommonProcess(
    const aclTensor* kv, const aclTensor* gamma, const aclTensor* cos, const aclTensor* sin, const aclTensor* index,
    aclTensor* kCacheRef, aclTensor* ckvCacheRef, const aclTensor* kRopeScale, const aclTensor* cKvScale,
    const aclTensor* kRopeOffset, const aclTensor* cKvOffset, const aclTensor* v, double epsilon, char* cacheMode,
    char* rotaryMode, char* quantMode, bool isOutputKv, const aclTensor* kRopeOut, const aclTensor* cKvOut,
    aclOpExecutor* executor)
{
    auto ret = CheckAiInfraKvRmsNormRopeCacheParams(
        kv, gamma, cos, sin, index, kCacheRef, ckvCacheRef, cacheMode, rotaryMode, quantMode, kRopeOut,
        cKvOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const aclTensor* kvFinal = l0op::Contiguous(kv, executor);
    CHECK_COND(kvFinal != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous kv failed.");
    const aclTensor* gammaFinal = l0op::Contiguous(gamma, executor);
    CHECK_COND(gammaFinal != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous gamma failed.");
    const aclTensor* cosFinal = l0op::Contiguous(cos, executor);
    CHECK_COND(cosFinal != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous cos failed.");
    const aclTensor* sinFinal = l0op::Contiguous(sin, executor);
    CHECK_COND(sinFinal != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous sin failed.");
    const aclTensor* indexFinal = l0op::Contiguous(index, executor);
    CHECK_COND(indexFinal != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous index failed.");

    aclTensor* kCacheFinal = nullptr;
    if (kCacheRef != nullptr) {
        kCacheFinal = executor->CreateView(
            kCacheRef, kCacheRef->GetViewShape(), kCacheRef->GetStorageShape(), kCacheRef->GetViewStrides(),
            kCacheRef->GetViewOffset());
        CHECK_COND(kCacheFinal != nullptr, ACLNN_ERR_INNER_NULLPTR, "CreateView for kCacheRef failed.");
    } else {
        // Preserve the vendor's one-dimensional absent-cache sentinel while
        // tying its lifetime to the executor, including failed preparation.
        kCacheFinal = executor->AllocTensor(op::Shape{0}, kvFinal->GetDataType());
        CHECK_COND(kCacheFinal != nullptr, ACLNN_ERR_INNER_NULLPTR, "Empty cache descriptor failed.");
    }
    aclTensor* ckvCacheFinal = executor->CreateView(
        ckvCacheRef, ckvCacheRef->GetViewShape(), ckvCacheRef->GetStorageShape(), ckvCacheRef->GetViewStrides(),
        ckvCacheRef->GetViewOffset());
    CHECK_COND(ckvCacheFinal != nullptr, ACLNN_ERR_INNER_NULLPTR, "CreateView for ckvCacheRef failed.");

    // 处理可选参数
    const aclTensor* kRopeScaleFinal = nullptr;
    if (kRopeScale != nullptr) {
        kRopeScaleFinal = l0op::Contiguous(kRopeScale, executor);
        CHECK_COND(kRopeScaleFinal != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous kRopeScale failed.");
    }
    const aclTensor* cKvScaleFinal = nullptr;
    if (cKvScale != nullptr) {
        cKvScaleFinal = l0op::Contiguous(cKvScale, executor);
        CHECK_COND(cKvScaleFinal != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous cKvScale failed.");
    }
    const aclTensor* kRopeOffsetFinal = nullptr;
    if (kRopeOffset != nullptr) {
        kRopeOffsetFinal = l0op::Contiguous(kRopeOffset, executor);
        CHECK_COND(kRopeOffsetFinal != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous kRopeOffset failed.");
    }
    const aclTensor* cKvOffsetFinal = nullptr;
    if (cKvOffset != nullptr) {
        cKvOffsetFinal = l0op::Contiguous(cKvOffset, executor);
        CHECK_COND(cKvOffsetFinal != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous cKvOffset failed.");
    }
    const aclTensor* vFinal = nullptr;
    if (v != nullptr) {
        vFinal = l0op::Contiguous(v, executor);
        CHECK_COND(vFinal != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous v failed.");
    }

    // out is always contiguous. passed directly to l0op.
    bool out = AiInfraKvRmsNormRopeCachel0op::AiInfraKvRmsNormRopeCache(
        kvFinal, gammaFinal, cosFinal, sinFinal, indexFinal, kCacheFinal, ckvCacheFinal, kRopeScaleFinal,
        cKvScaleFinal, kRopeOffsetFinal, cKvOffsetFinal, vFinal, epsilon, cacheMode, rotaryMode, quantMode,
        isOutputKv, kRopeOut, cKvOut, executor);
    CHECK_RET(out, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

aclnnStatus ifOwnedAiInfraKvRmsNormRopeCacheGetWorkspaceSize(
    const aclTensor* kv, const aclTensor* gamma, const aclTensor* cos, const aclTensor* sin, const aclTensor* index,
    aclTensor* kCacheRef, aclTensor* ckvCacheRef, const aclTensor* kRopeScaleOptional,
    const aclTensor* ckvScaleOptional, const aclTensor* kRopeOffsetOptional, const aclTensor* cKvOffsetOptional,
    const aclTensor* vOptional, double epsilon, char* cacheModeOptional, char* rotaryModeOptional,
    char* quantModeOptional, bool isOutputKv, aclTensor* kRopeOut, aclTensor* cKvOut, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        ifOwnedAiInfraKvRmsNormRopeCache,
        DFX_IN(
            kv, gamma, cos, sin, index, kCacheRef, ckvCacheRef, kRopeScaleOptional, ckvScaleOptional,
            kRopeOffsetOptional, cKvOffsetOptional, vOptional),
        DFX_OUT(kRopeOut, cKvOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    auto ret = AiInfraKvRmsNormRopeCacheCommonProcess(
        kv, gamma, cos, sin, index, kCacheRef, ckvCacheRef, kRopeScaleOptional, ckvScaleOptional, kRopeOffsetOptional,
        cKvOffsetOptional, vOptional, epsilon, cacheModeOptional, rotaryModeOptional, quantModeOptional, isOutputKv,
        kRopeOut, cKvOut, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus ifOwnedAiInfraKvRmsNormRopeCache(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(ifOwnedAiInfraKvRmsNormRopeCache);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif
