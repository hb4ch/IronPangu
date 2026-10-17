#pragma once
#include "resident_tensor.h"
#include <cmath>
#include <aclnn_ai_infra_fused_infer_attention_sink.h>
#include <aclnn_ai_infra_fused_infer_attention_sink_metadata.h>
namespace inferfabric::resident {
inline void window_metadata(aclTensor *ends,aclTensor *lengths,uint32_t window,
    int64_t cube,int64_t vector,aclTensor *output) {
  auto &context=*prepared::active;char layout[]="TND",cache_layout[]="BnBsH";
  uint64_t bytes=0;aclOpExecutor *executor=nullptr;
  prepared::check(prepared::query("attention_metadata",false,
      aclnnAiInfraFusedInferAttentionSinkMetadataGetWorkspaceSize,ends,lengths,
      6,1,512,512,1,4,int64_t(window)-1,0,layout,cache_layout,0,128,64,16,cube,vector,
      output,&bytes,&executor),"window metadata prepare");
  prepared::check(aclnnAiInfraFusedInferAttentionSinkMetadata(context.workspace(bytes),bytes,executor,context.stream),"window metadata launch");
}
inline void window_attention(aclTensor *query,aclTensorList *keys,aclTensorList *values,
    aclTensor *mask,aclTensor *ends,aclTensor *lengths,aclTensor *tables,
    aclTensor *query_rope,aclTensor *key_rope,aclTensor *metadata,
    aclTensor *sink,aclTensor *sink_rope,uint32_t window,aclTensor *output) {
  auto &context=*prepared::active;char layout[]="TND";
  uint64_t bytes=0;aclOpExecutor *executor=nullptr;
  prepared::check(prepared::query("aclnnAiInfraFusedInferAttentionSink",false,
      aclnnAiInfraFusedInferAttentionSinkGetWorkspaceSize,query,keys,values,
      nullptr,mask,ends,lengths,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
      tables,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
      query_rope,key_rope,nullptr,nullptr,metadata,sink,sink_rope,sink,
      6,1/std::sqrt(192.0),int64_t(window)-1,0,layout,1,4,0,16,0,false,0,0,0,0,
      output,nullptr,&bytes,&executor),"window attention prepare");
  prepared::check(aclnnAiInfraFusedInferAttentionSink(context.workspace(bytes),bytes,executor,context.stream),"window attention launch");
}
}
