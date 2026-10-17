#pragma once
#include "resident_tensor.h"
#include <cmath>
#include <aclnn_ai_infra_kv_quant_sparse_flash_attention.h>
namespace inferfabric::resident {
inline void packed_attention(aclTensor *query,aclTensor *cache,aclTensor *indices,
    aclTensor *tables,aclTensor *query_ends,aclTensor *kv_lengths,
    aclTensor *sink,aclTensor *sink_values,aclTensor *output) {
  auto &context=*prepared::active;
  char layout[]="TND",cache_layout[]="PA_BSND";
  uint64_t bytes=0;aclOpExecutor *executor=nullptr;
  prepared::check(prepared::query("aclnnAiInfraKvQuantSparseFlashAttention",false,
      aclnnAiInfraKvQuantSparseFlashAttentionGetWorkspaceSize,
      query,cache,cache,indices,nullptr,nullptr,tables,query_ends,kv_lengths,sink,sink_values,
      1/std::sqrt(192.0),2,2,1,layout,cache_layout,3,INT64_MAX,INT64_MAX,2,1,128,64,
      output,&bytes,&executor),"packed attention prepare");
  prepared::check(aclnnAiInfraKvQuantSparseFlashAttention(context.workspace(bytes),bytes,executor,context.stream),
      "packed attention launch");
}
}
