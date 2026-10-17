#pragma once
#include "resident_tensor.h"
#include <aclnnop/aclnn_rotary_position_embedding.h>

namespace inferfabric::resident {
inline Tensor column_slice(const Tensor &input, int64_t rows, int64_t start, int64_t width) {
  if(!prepared::active || input.dtype()!=ACL_BF16 || rows<=0 || input.size()%rows || start<0 || width<=0 ||
      start>int64_t(input.size()/rows) || width>int64_t(input.size()/rows)-start)
    throw std::runtime_error("column slice contract");
  auto &context=*prepared::active;
  int64_t columns=input.size()/rows;
  auto *view=context.strided_descriptor(static_cast<uint16_t *>(input.data())+start,
      input.size()-start,{rows,width},{columns,1},ACL_BF16);
  openpangu::DeviceTensor out({rows,width},ACL_BF16,2);
  uint64_t bytes=0;aclOpExecutor *executor=nullptr;
  prepared::check(prepared::query("column_slice",false,aclnnCastGetWorkspaceSize,
      view,ACL_BF16,out.tensor,&bytes,&executor),"column slice prepare");
  prepared::check(aclnnCast(context.workspace(bytes),bytes,executor,context.stream),"column slice launch");
  return Tensor::alias(out,ACL_BF16);
}

// The caller supplies a device coefficient row for the current absolute position.
// This routine cannot upload, read back, or silently convert any operand.
inline Tensor rope_bf16(const Tensor &input, aclTensor *cosine, aclTensor *sine) {
  if(!prepared::active || input.dtype()!=ACL_BF16 || !input.size() || input.size()%64 || !cosine || !sine)
    throw std::runtime_error("BF16 rotary contract");
  auto &context=*prepared::active;
  int64_t heads=input.size()/64;
  openpangu::DeviceTensor out({1,heads,1,64},ACL_BF16,2);
  auto *view=context.descriptor(input.data(),{1,heads,1,64},ACL_BF16);
  uint64_t bytes=0;aclOpExecutor *executor=nullptr;
  prepared::check(prepared::query("aclnnRotaryPositionEmbedding",false,aclnnRotaryPositionEmbeddingGetWorkspaceSize,
      view,cosine,sine,0,out.tensor,&bytes,&executor),"BF16 rotary prepare");
  prepared::check(aclnnRotaryPositionEmbedding(context.workspace(bytes),bytes,executor,context.stream),"BF16 rotary launch");
  return Tensor::alias(out,ACL_BF16);
}
} // namespace inferfabric::resident
