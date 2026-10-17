#pragma once
#include "resident_tensor.h"
#include "../vendor/omni_opapi/aclnn_ai_infra_kv_rms_norm_rope_cache.h"
namespace inferfabric::resident {
inline void normalized_cache_append(const Tensor &input,aclTensor *gamma,aclTensor *cosine,
    aclTensor *sine,aclTensor *positions,aclTensor *cache,uint32_t rows,bool packed) {
  if(!rows || rows>4096 || input.size()!=size_t(rows)*576 || input.dtype()!=ACL_BF16)
    throw std::runtime_error("batch cache input contract");
  auto &context=*prepared::active;
  auto *kv=context.descriptor(input.data(),{rows,1,1,576},ACL_BF16);
  // Required vendor outputs, even though the consumer only uses the cache.
  openpangu::DeviceTensor rope({rows,1,1,64},ACL_BF16,2),latent({rows,1,1,512},ACL_BF16,2);
  char mode[]="PA",rotary[]="half",none[]="none",c8[]="pertile128";
  uint64_t bytes=0;aclOpExecutor *executor=nullptr;
  prepared::check(prepared::query("aclnnAiInfraKvRmsNormRopeCache",false,
      ifOwnedAiInfraKvRmsNormRopeCacheGetWorkspaceSize,kv,gamma,cosine,sine,positions,
      nullptr,cache,nullptr,nullptr,nullptr,nullptr,nullptr,1e-5,mode,rotary,packed?c8:none,
      true,rope.tensor,latent.tensor,&bytes,&executor),"batch cache prepare");
  context.require_stream(context.stream);
  prepared::check(ifOwnedAiInfraKvRmsNormRopeCache(context.workspace(bytes),bytes,executor,context.stream),"batch cache append");
}
}
