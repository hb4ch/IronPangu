#pragma once
#include "resident_tensor.h"
#include <aclnn_ai_infra_fused_causal_conv1d.h>
namespace inferfabric::resident {
// Segmented causal convolution: starts[N+1], slots[N] and positions[N]
// describe independent histories in the shared state tensor. Scratch must not
// alias input: the vendor's inplace update happens after a cross-core barrier.
inline void causal_convolution(aclTensor *input,aclTensor *weight,aclTensor *state,
    aclTensor *starts,aclTensor *slots,aclTensor *positions,aclTensor *scratch,uint32_t rows) {
  if(!rows)throw std::runtime_error("empty convolution batch");
  auto &context=*prepared::active;
  uint64_t bytes=0;aclOpExecutor *executor=nullptr;
  prepared::check(prepared::query("aclnnAiInfraFusedCausalConv1d",false,
      aclnnAiInfraFusedCausalConv1dGetWorkspaceSize,input,weight,state,starts,slots,
      nullptr,nullptr,nullptr,positions,nullptr,nullptr,nullptr,
      0,-1,0,rows,1,128,1,true,scratch,&bytes,&executor),"segmented convolution prepare");
  prepared::check(aclnnAiInfraFusedCausalConv1d(context.workspace(bytes),bytes,executor,context.stream),
      "segmented convolution launch");
}
}
