#pragma once
#include "resident_tensor.h"
#include <aclnnop/aclnn_batch_matmul.h>

namespace inferfabric::resident {
inline Tensor head_matmul(aclTensor *input,aclTensor *weight,uint32_t heads,uint32_t tokens,uint32_t width) {
  auto &context=*prepared::active;
  openpangu::DeviceTensor out({heads,tokens,width},ACL_BF16,2);
  uint64_t bytes=0;aclOpExecutor *executor=nullptr;
  prepared::check(prepared::query("aclnnBatchMatMul",true,aclnnBatchMatMulGetWorkspaceSize,
      input,weight,out.tensor,0,&bytes,&executor),"head matmul prepare");
  context.require_stream(context.stream);
  prepared::check(aclnnBatchMatMul(context.workspace(bytes),bytes,executor,context.stream),"head matmul");
  return Tensor::alias(out,ACL_BF16);
}
inline Tensor head_to_token(const Tensor &input,uint32_t heads,uint32_t tokens,uint32_t width) {
  if(!heads || !tokens || !width || input.dtype()!=ACL_BF16 || input.size()!=size_t(tokens)*heads*width)
    throw std::runtime_error("head layout storage");
  if(tokens==1)return input;
  auto &context=*prepared::active;
  auto *source=context.strided_descriptor(input.data(),input.size(),
      {tokens,heads,width},{width,int64_t(tokens)*width,1},ACL_BF16);
  openpangu::DeviceTensor out({tokens,heads,width},ACL_BF16,2);
  uint64_t bytes=0;aclOpExecutor *executor=nullptr;
  prepared::check(prepared::query("head_to_token",true,aclnnCastGetWorkspaceSize,
      source,ACL_BF16,out.tensor,&bytes,&executor),"head layout prepare");
  context.require_stream(context.stream);
  prepared::check(aclnnCast(context.workspace(bytes),bytes,executor,context.stream),"head layout");
  return Tensor::alias(out,ACL_BF16);
}
}
