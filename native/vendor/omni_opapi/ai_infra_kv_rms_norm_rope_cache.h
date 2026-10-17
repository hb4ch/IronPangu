/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_COMMON_INC_LEVEL0_OP_AI_INFRA_KV_RMS_NORM_ROPE_CACHE_H_
#define OP_API_COMMON_INC_LEVEL0_OP_AI_INFRA_KV_RMS_NORM_ROPE_CACHE_H_

#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

namespace AiInfraKvRmsNormRopeCachel0op {
const aclTensor* AiInfraKvRmsNormRopeCache(
    const aclTensor* kv, const aclTensor* gamma, const aclTensor* cos, const aclTensor* sin, const aclTensor* index,
    aclTensor* kCache, aclTensor* ckvCache, const aclTensor* kRopeScale, const aclTensor* cKvScale,
    const aclTensor* kRopeOffset, const aclTensor* cKvOffset, const aclTensor* v, double epsilon, char* cacheMode,
    char* rotaryMode, char* quantMode, bool isOutputKv, const aclTensor* kRopeOut, const aclTensor* cKvOut,
    aclOpExecutor* executor);
}

#endif // OP_API_COMMON_INC_LEVEL0_OP_AI_INFRA_KV_RMS_NORM_ROPE_CACHE_H_