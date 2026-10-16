#pragma once
#if defined(IF_REQUIRE_DEVICE_RESIDENCY) && !defined(IF_RESIDENT_TRANSFER_GUARD_ACTIVE)
#error "Device-resident target requires transfer guard before all operator headers"
#endif
// Native OpenPangu building blocks. Caller owns stream and fences before destruction.
#include <acl/acl.h>
#include "prepared_execution.h"
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_moe_gating_top_k.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace openpangu {
inline void checked(int status,const char *op) {
  if(status)throw std::runtime_error(std::string(op)+": "+std::to_string(status));
}
struct FloatFixture {
  std::vector<int64_t> shape;std::vector<float> values;
  explicit FloatFixture(const std::string &path) {
    std::ifstream f(path,std::ios::binary);char magic[8];uint64_t rank=0,count=1;
    f.read(magic,8);f.read(reinterpret_cast<char*>(&rank),8);
    if(!f || std::memcmp(magic,"IFF32T01",8) || rank<1 || rank>4)throw std::runtime_error("float fixture header");
    shape.resize(rank);f.read(reinterpret_cast<char*>(shape.data()),rank*8);
    for(auto d:shape){if(d<=0 || uint64_t(d)>67108864/count)throw std::runtime_error("float fixture bounds");count*=d;}
    values.resize(count);f.read(reinterpret_cast<char*>(values.data()),count*4);
    if(!f || f.peek()!=EOF)throw std::runtime_error("float fixture size");
    for(float v:values)if(!std::isfinite(v))throw std::runtime_error("non-finite float fixture");
  }
};
struct DeviceTensor {
  void *data=nullptr;aclTensor *tensor=nullptr;size_t bytes=0;
  inferfabric::prepared::Context *owner=nullptr;
  std::shared_ptr<void> allocation;
  aclDataType type;
  DeviceTensor(std::vector<int64_t> shape,aclDataType dtype,size_t width):bytes(width),type(dtype) {
    if(shape.empty())throw std::runtime_error("empty tensor shape");
    std::vector<int64_t> strides(shape.size(),1);
    for(auto d:shape){if(d<=0 || uint64_t(d)>SIZE_MAX/bytes)throw std::runtime_error("tensor bounds");bytes*=d;}
    for(size_t j=shape.size()-1;j>0;--j)strides[j-1]=strides[j]*shape[j];
    owner=inferfabric::prepared::active;
    if(owner){data=owner->acquire(bytes);auto *context=owner;allocation=std::shared_ptr<void>(data,[context](void *p){context->release(p);});tensor=owner->descriptor(data,shape,dtype);return;}
    checked(aclrtMalloc(&data,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"tensor allocation");
    tensor=aclCreateTensor(shape.data(),shape.size(),dtype,strides.data(),0,ACL_FORMAT_ND,shape.data(),shape.size(),data);
    if(!tensor){aclrtFree(data);data=nullptr;throw std::runtime_error("tensor descriptor");}
  }
  DeviceTensor(const DeviceTensor&)=delete;DeviceTensor& operator=(const DeviceTensor&)=delete;
  ~DeviceTensor(){if(owner){return;}if(tensor)aclDestroyTensor(tensor);if(data)aclrtFree(data);}
  template<class T> void upload(const std::vector<T>& v){if(v.size()*sizeof(T)!=bytes)throw std::runtime_error("upload size");if(owner){owner->upload(data,v.data(),bytes);return;}checked(aclrtMemcpy(data,bytes,v.data(),bytes,ACL_MEMCPY_HOST_TO_DEVICE),"tensor upload");}
  template<class T> std::vector<T> read() const {std::vector<T> v(bytes/sizeof(T));checked(aclrtMemcpy(v.data(),bytes,data,bytes,ACL_MEMCPY_DEVICE_TO_HOST),"tensor download");return v;}
};
using Launcher=aclnnStatus(*)(void*,uint64_t,aclOpExecutor*,aclrtStream);
struct Operation {
  aclOpExecutor *executor=nullptr;uint64_t bytes=0;void *workspace=nullptr;uint64_t capacity=0;
  Operation()=default;Operation(const Operation&)=delete;
  void allocate(){capacity=std::max(uint64_t(64),bytes);checked(aclrtMalloc(&workspace,capacity,ACL_MEM_MALLOC_HUGE_FIRST),"operator workspace");}
  void run(Launcher launch,aclrtStream stream){if(bytes>capacity)throw std::runtime_error("workspace grew");checked(launch(workspace,bytes,executor,stream),"operator launch");executor=nullptr;}
  ~Operation(){if(executor)aclDestroyAclOpExecutor(executor);if(workspace)aclrtFree(workspace);}
};
// Replicated FP32 gate. Bias affects selection only; unbiased sigmoid scores
// normalize over the selected eight experts, then scale by 2.5.
struct Router {
  int64_t batch,hidden,experts,topk;
  aclrtStream stream;
  DeviceTensor input,weight,bias,logits,scores,ids,normalized;
  Operation projection,selection;
  Router(int64_t b,int64_t h,int64_t e,int64_t k,aclrtStream s):batch(b),hidden(h),experts(e),topk(k),stream(s),
    input({b,h},ACL_FLOAT,4),weight({h,e},ACL_FLOAT,4),bias({e},ACL_FLOAT,4),logits({b,e},ACL_FLOAT,4),
    scores({b,k},ACL_FLOAT,4),ids({b,k},ACL_INT32,4),normalized({b,e},ACL_FLOAT,4) {
    if(e!=256 || k!=8 || h!=2560 || b<1 || b>16)throw std::runtime_error("unsupported 92B router shape");
    query_projection();projection.allocate();query_selection();selection.allocate();
  }
  void load(const FloatFixture &w,const FloatFixture &correction) {
    if(w.shape!=std::vector<int64_t>{experts,hidden} || correction.shape!=std::vector<int64_t>{experts})throw std::runtime_error("router checkpoint shape");
    std::vector<float> transposed(w.values.size());
    for(int64_t e=0;e<experts;++e)for(int64_t h=0;h<hidden;++h)transposed[h*experts+e]=w.values[e*hidden+h];
    weight.upload(transposed);bias.upload(correction.values);
  }
  void query_projection(){checked(aclnnMatmulGetWorkspaceSize(input.tensor,weight.tensor,logits.tensor,0,&projection.bytes,&projection.executor),"router matmul prepare");}
  void query_selection(){checked(aclnnMoeGatingTopKGetWorkspaceSize(logits.tensor,bias.tensor,topk,1,1,1,0,1,true,2.5,1e-20,scores.tensor,ids.tensor,normalized.tensor,&selection.bytes,&selection.executor),"router topk prepare");}
  void select(){if(!selection.executor)query_selection();selection.run(aclnnMoeGatingTopK,stream);}
  void launch(){if(!projection.executor)query_projection();projection.run(aclnnMatmul,stream);select();}
};
} // namespace openpangu
