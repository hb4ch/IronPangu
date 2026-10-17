#pragma once
#include "resident_tensor.h"
#include "page_binding.h"
#include <aclnnop/aclnn_index_copy.h>
#include <aclnn_lightning_indexer_enhance.h>
#include <aclnn_ai_infra_quant_lightning_indexer.h>
#include <dlfcn.h>

namespace inferfabric::resident {
struct SparseIndexApi {
  void *handle=nullptr;
  decltype(&aclnnLightningIndexerEnhanceGetWorkspaceSize) prepare=nullptr;
  decltype(&aclnnLightningIndexerEnhance) launch=nullptr;
  SparseIndexApi() {
    handle=dlopen(OMNI_TRAINING_OPAPI_PATH,RTLD_NOW|RTLD_LOCAL);
    if(!handle)throw std::runtime_error(dlerror());
    prepare=reinterpret_cast<decltype(prepare)>(dlsym(handle,"aclnnLightningIndexerEnhanceGetWorkspaceSize"));
    launch=reinterpret_cast<decltype(launch)>(dlsym(handle,"aclnnLightningIndexerEnhance"));
    if(!prepare || !launch){dlclose(handle);handle=nullptr;throw std::runtime_error("missing sparse index API");}
  }
  ~SparseIndexApi(){if(handle)dlclose(handle);}
};
inline SparseIndexApi &sparse_index_api(){static SparseIndexApi api;return api;}

struct IndexState {
  uint32_t capacity,residents;
  bool shared,quantized;
  state::ResidentPages pages;
  openpangu::DeviceTensor keys,table,query_length;
  std::unique_ptr<openpangu::DeviceTensor> scales;
  IndexState(uint32_t cap,uint32_t slots,bool shared_pages,bool quant=false,uint32_t max_queries=1)
      :capacity(cap),residents(slots),shared(shared_pages),quantized(quant),pages(64,cap,slots,shared_pages),
       keys({int64_t(cap)*slots/64,64,1,128},quant?ACL_INT8:ACL_BF16,quant?1:2),table({1,cap/64},ACL_INT32,4),
       query_length({max_queries},ACL_INT32,4) {
    if(!prepared::active)throw std::runtime_error("index state requires prepared context");
    if(quantized){scales=std::make_unique<openpangu::DeviceTensor>(std::vector<int64_t>{int64_t(cap)*slots/64,64,1},ACL_FLOAT16,2);prepared::active->zero(scales->data,scales->bytes);}
    prepared::active->zero(keys.data,keys.bytes);upload_table();
    std::vector<int32_t> lengths(max_queries);
    for(uint32_t row=0;row<max_queries;++row)lengths[row]=row+1;
    control_upload(query_length,lengths);
  }
  void upload_table(){
    std::vector<int32_t> ids(capacity/64,0);
    std::copy(pages.pages().begin(),pages.pages().end(),ids.begin());control_upload(table,ids);
  }
  void select(uint32_t slot){pages.select(slot);upload_table();}
  void bind(const std::vector<int32_t> &ids){pages.bind(ids);upload_table();}
  void clear(uint32_t slot,bool active){
    for(auto page:pages.pages(slot)) {
      size_t bytes=64*128*(quantized?1:2);
      prepared::active->zero(static_cast<uint8_t *>(keys.data)+size_t(page)*bytes,bytes);
      if(quantized)prepared::active->zero(static_cast<uint8_t *>(scales->data)+size_t(page)*64*2,64*2);
    }
    if(shared)pages.release(slot);
    if(active)upload_table();
  }
  void append(const Tensor &key,aclTensor *position){
    if(key.dtype()!=ACL_BF16 || key.size()!=128)throw std::runtime_error("index key contract");
    auto &context=*prepared::active;
    auto *cache=context.descriptor(keys.data,{int64_t(capacity)*residents,128},ACL_BF16);
    auto *source=context.descriptor(key.data(),{1,128},ACL_BF16);
    uint64_t bytes=0;aclOpExecutor *executor=nullptr;
    prepared::check(prepared::query("index_cache_append",false,aclnnInplaceIndexCopyGetWorkspaceSize,
        cache,0,position,source,&bytes,&executor),"index cache append prepare");
    prepared::check(aclnnInplaceIndexCopy(context.workspace(bytes),bytes,executor,context.stream),"index cache append");
  }
  Tensor select_indices(const Tensor &query,const Tensor &weights,aclTensor *length){
    if(query.dtype()!=ACL_BF16 || query.size()!=24*128 || weights.dtype()!=ACL_BF16 || weights.size()!=24)
      throw std::runtime_error("sparse index operands");
    auto &context=*prepared::active;auto &api=sparse_index_api();
    openpangu::DeviceTensor out({1,1,2048},ACL_INT32,4);
    uint64_t bytes=0;aclOpExecutor *executor=nullptr;
    prepared::check(prepared::query("aclnnLightningIndexerEnhance",false,api.prepare,
        context.descriptor(query.data(),{1,24,128},ACL_BF16),keys.tensor,
        context.descriptor(weights.data(),{1,24},ACL_BF16),context.descriptor(query_length.data,{1},ACL_INT32),length,table.tensor,
        "TND","PA_BSND",2048,3,INT64_MAX,INT64_MAX,false,1,0,out.tensor,nullptr,&bytes,&executor),"sparse index prepare");
    prepared::check(api.launch(context.workspace(bytes),bytes,executor,context.stream),"sparse index launch");
    return Tensor::alias(out,ACL_INT32);
  }
  void append_quantized(const Tensor &key,const Tensor &scale,aclTensor *position){
    if(!quantized || key.dtype()!=ACL_INT8 || key.size()!=128 || scale.dtype()!=ACL_FLOAT16 || scale.size()!=1)
      throw std::runtime_error("quantized index key contract");
    auto &context=*prepared::active;
    auto append=[&](openpangu::DeviceTensor &dest,const Tensor &src,int64_t width,aclDataType dtype){
      uint64_t bytes=0;aclOpExecutor *executor=nullptr;
      prepared::check(prepared::query("index_cache_append",false,aclnnInplaceIndexCopyGetWorkspaceSize,
        context.descriptor(dest.data,{int64_t(capacity)*residents,width},dtype),0,position,
        context.descriptor(src.data(),{1,width},dtype),&bytes,&executor),"quant index append prepare");
      prepared::check(aclnnInplaceIndexCopy(context.workspace(bytes),bytes,executor,context.stream),"quant index append");
    };
    append(keys,key,128,ACL_INT8);append(*scales,scale,1,ACL_FLOAT16);
  }
  Tensor select_quantized(const Tensor &query,const Tensor &query_scale,const Tensor &weights,aclTensor *length,uint32_t rows){
    if(!rows || rows>query_length.bytes/sizeof(int32_t) || !quantized || query.dtype()!=ACL_INT8 || query.size()!=size_t(rows)*3072 || query_scale.dtype()!=ACL_FLOAT16 || query_scale.size()!=size_t(rows)*24 || weights.dtype()!=ACL_FLOAT16 || weights.size()!=size_t(rows)*24)
      throw std::runtime_error("quantized sparse index operands");
    auto &context=*prepared::active;
    openpangu::DeviceTensor out({rows,1,2048},ACL_INT32,4);
    uint64_t bytes=0;aclOpExecutor *executor=nullptr;
    prepared::check(prepared::query("aclnnAiInfraQuantLightningIndexer",false,aclnnAiInfraQuantLightningIndexerGetWorkspaceSize,
      context.descriptor(query.data(),{rows,24,128},ACL_INT8),keys.tensor,
      context.descriptor(weights.data(),{rows,24},ACL_FLOAT16),context.descriptor(query_scale.data(),{rows,24},ACL_FLOAT16),
      scales->tensor,context.descriptor(static_cast<int32_t *>(query_length.data)+rows-1,{1},ACL_INT32),length,table.tensor,0,0,"TND","PA_BSND",2048,3,INT64_MAX,INT64_MAX,
      out.tensor,&bytes,&executor),"quant sparse index prepare");
    prepared::check(aclnnAiInfraQuantLightningIndexer(context.workspace(bytes),bytes,executor,context.stream),"quant sparse index launch");
    return Tensor::alias(out,ACL_INT32);
  }
};
} // namespace inferfabric::resident
