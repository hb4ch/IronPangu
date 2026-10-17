#pragma once
#include "resident_tensor.h"
#include <aclnn_ai_infra_quant_lightning_indexer.h>
namespace inferfabric::resident {
inline void quant_index(aclTensor *query,aclTensor *keys,aclTensor *weights,
    aclTensor *query_scales,aclTensor *key_scales,aclTensor *query_ends,
    aclTensor *kv_lengths,aclTensor *tables,aclTensor *output) {
  auto &context=*prepared::active;
  uint64_t bytes=0;aclOpExecutor *executor=nullptr;
  prepared::check(prepared::query("aclnnAiInfraQuantLightningIndexer",false,
      aclnnAiInfraQuantLightningIndexerGetWorkspaceSize,query,keys,weights,query_scales,
      key_scales,query_ends,kv_lengths,tables,0,0,"TND","PA_BSND",2048,3,INT64_MAX,INT64_MAX,
      output,&bytes,&executor),"quant sparse index prepare");
  prepared::check(aclnnAiInfraQuantLightningIndexer(context.workspace(bytes),bytes,executor,context.stream),
      "quant sparse index launch");
}
}
