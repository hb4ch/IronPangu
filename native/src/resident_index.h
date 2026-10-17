#pragma once
#include "resident_tensor.h"
#include "page_binding.h"
#include <aclnnop/aclnn_index_copy.h>
#include <aclnn_lightning_indexer_enhance.h>
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
  bool shared;
  state::ResidentPages pages;
  openpangu::DeviceTensor keys,table,query_length;
  IndexState(uint32_t cap,uint32_t slots,bool shared_pages)
      :capacity(cap),residents(slots),shared(shared_pages),pages(64,cap,slots,shared_pages),
       keys({int64_t(cap)*slots/64,64,1,128},ACL_BF16,2),table({1,cap/64},ACL_INT32,4),
       query_length({1},ACL_INT32,4) {
    if(!prepared::active)throw std::runtime_error("index state requires prepared context");
    prepared::active->zero(keys.data,keys.bytes);upload_table();
    control_upload(query_length,std::vector<int32_t>{1});
  }
  void upload_table(){
    std::vector<int32_t> ids(capacity/64,0);
    std::copy(pages.pages().begin(),pages.pages().end(),ids.begin());control_upload(table,ids);
  }
  void select(uint32_t slot){pages.select(slot);upload_table();}
  void bind(const std::vector<int32_t> &ids){pages.bind(ids);upload_table();}
  void clear(uint32_t slot,bool active){
    for(auto page:pages.pages(slot))prepared::active->zero(static_cast<uint8_t *>(keys.data)+size_t(page)*64*128*2,64*128*2);
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
        context.descriptor(weights.data(),{1,24},ACL_BF16),query_length.tensor,length,table.tensor,
        "TND","PA_BSND",2048,3,INT64_MAX,INT64_MAX,false,1,0,out.tensor,nullptr,&bytes,&executor),"sparse index prepare");
    prepared::check(api.launch(context.workspace(bytes),bytes,executor,context.stream),"sparse index launch");
    return Tensor::alias(out,ACL_INT32);
  }
};
} // namespace inferfabric::resident
