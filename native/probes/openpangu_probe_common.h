#pragma once
// Native W8A8 qualification with real checkpoint weights and synthetic inputs.
// Tests dynamic quantization, per-token/per-channel scales and stable-address graph replay.
#include <acl/acl.h>
#include <aclnnop/aclnn_dynamic_quant.h>
#include <aclnnop/aclnn_quant_matmul_v5.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
void check(int code, const char *what) {
  if (code) { const char *detail=aclGetRecentErrMsg(); if(detail) std::cerr<<detail<<"\n"; }
  if (code) throw std::runtime_error(std::string(what)+": "+std::to_string(code));
}
float fp32(uint16_t b) { uint32_t u=uint32_t(b)<<16;float f;std::memcpy(&f,&u,4);return f; }
uint16_t bf16(float f) {uint32_t u;std::memcpy(&u,&f,4);return uint16_t((u+0x7fff+((u>>16)&1))>>16);}
struct Tensor { void *data=nullptr;aclTensor *t=nullptr;size_t bytes=0; };
struct Runner {
  int64_t n=0,k=0,m=4;
  std::vector<int8_t> weight;
  std::vector<float> scale;
  std::vector<Tensor> tensors;
  aclrtStream stream=nullptr;bool owns_stream=true;aclmdlRI graph=nullptr;
  aclOpExecutor *dq=nullptr,*mm=nullptr;
  uint64_t dq_bytes=0,mm_bytes=0;void *workspace=nullptr;
  double max_error=0,max_quant_error=0;size_t checked=0;
  explicit Runner(const char *path,int batch):m(batch) {
    std::ifstream f(path,std::ios::binary);char magic[8];
    f.read(magic,8);f.read(reinterpret_cast<char*>(&n),8);f.read(reinterpret_cast<char*>(&k),8);
    if (!f || std::memcmp(magic,"IFW8A801",8) || n<=0 || k<=0 || n>65536 || k>65536 || uint64_t(n)*k>256*1024*1024 || m<1 || m>16)
      throw std::runtime_error("invalid fixture dimensions/header");
    weight.resize(n*k);scale.resize(n);
    f.read(reinterpret_cast<char*>(weight.data()),weight.size());f.read(reinterpret_cast<char*>(scale.data()),n*4);
    if (!f || f.peek()!=EOF) throw std::runtime_error("fixture size");
    for (auto s:scale) if (!std::isfinite(s) || s<=0) throw std::runtime_error("invalid scale");
  }
  void tensor(std::vector<int64_t> shape,aclDataType dtype,size_t width) {
    Tensor t;t.bytes=width;std::vector<int64_t> strides(shape.size(),1);
    for (auto d:shape) t.bytes*=d;
    for (size_t j=shape.size()-1;j>0;--j)strides[j-1]=strides[j]*shape[j];
    check(aclrtMalloc(&t.data,t.bytes,ACL_MEM_MALLOC_HUGE_FIRST),"tensor allocate");
    t.t=aclCreateTensor(shape.data(),shape.size(),dtype,strides.data(),0,ACL_FORMAT_ND,shape.data(),shape.size(),t.data);
    if(!t.t){aclrtFree(t.data);throw std::runtime_error("tensor descriptor");}
    tensors.push_back(t);
  }
  void upload(size_t id,const void *p) {check(aclrtMemcpy(tensors[id].data,tensors[id].bytes,p,tensors[id].bytes,ACL_MEMCPY_HOST_TO_DEVICE),"upload");}
  template<class T> std::vector<T> read(size_t id) {
    std::vector<T> v(tensors[id].bytes/sizeof(T));
    check(aclrtMemcpy(v.data(),tensors[id].bytes,tensors[id].data,tensors[id].bytes,ACL_MEMCPY_DEVICE_TO_HOST),"read");return v;
  }
  void prepare(aclrtStream shared=nullptr) {
    owns_stream=shared==nullptr;stream=shared;
    if(owns_stream)check(aclrtCreateStream(&stream),"stream");
    tensor({m,k},ACL_BF16,2);tensor({m,k},ACL_INT8,1);tensor({m},ACL_FLOAT,4);
    tensor({n,k},ACL_INT8,1);tensor({n},ACL_FLOAT,4);tensor({m,n},ACL_BF16,2);
    upload(3,weight.data());upload(4,scale.data());
    check(aclnnDynamicQuantGetWorkspaceSize(tensors[0].t,nullptr,tensors[1].t,tensors[2].t,&dq_bytes,&dq),"dynamic quant prepare");
    check(aclnnQuantMatmulV5GetWorkspaceSize(tensors[1].t,tensors[3].t,tensors[2].t,tensors[4].t,nullptr,nullptr,nullptr,nullptr,nullptr,false,true,0,tensors[5].t,&mm_bytes,&mm),"W8A8 prepare");
    check(aclrtMalloc(&workspace,std::max(uint64_t(64),std::max(dq_bytes,mm_bytes)),ACL_MEM_MALLOC_HUGE_FIRST),"workspace");
  }
  void launch() {
    // This CANN build rejects repeatable DynamicQuant executors. Prepare only
    // during warmup/capture; graph replays call neither workspace-size API.
    if(!dq) {
      uint64_t bytes=0;
      check(aclnnDynamicQuantGetWorkspaceSize(tensors[0].t,nullptr,tensors[1].t,tensors[2].t,&bytes,&dq),"dynamic quant reprepare");
      if(bytes>dq_bytes)throw std::runtime_error("dynamic quant workspace changed");
    }
    if(!mm) {
      uint64_t bytes=0;
      check(aclnnQuantMatmulV5GetWorkspaceSize(tensors[1].t,tensors[3].t,tensors[2].t,tensors[4].t,nullptr,nullptr,nullptr,nullptr,nullptr,false,true,0,tensors[5].t,&bytes,&mm),"W8A8 reprepare");
      if(bytes>mm_bytes)throw std::runtime_error("W8A8 workspace changed");
    }
    check(aclnnDynamicQuant(workspace,dq_bytes,dq,stream),"dynamic quant");dq=nullptr;
    check(aclnnQuantMatmulV5(workspace,mm_bytes,mm,stream),"W8A8");mm=nullptr;
  }
  std::vector<uint16_t> input(int variant) {
    std::vector<uint16_t> x(m*k);
    for(int64_t r=0;r<m;++r)for(int64_t c=0;c<k;++c)
      x[r*k+c]=bf16(variant==2?0.f:float(((c*17+r*11+variant*23)%251)-125)/float(37+variant*31));
    upload(0,x.data());return x;
  }
  std::vector<uint16_t> verify(const std::vector<uint16_t> &x) {
    auto q=read<int8_t>(1);auto ds=read<float>(2);auto y=read<uint16_t>(5);
    for(int64_t r=0;r<m;++r) {
      float amax=0;for(int64_t c=0;c<k;++c)amax=std::max(amax,std::abs(fp32(x[r*k+c])));
      if(!std::isfinite(ds[r]) || ds[r]<0 || (amax>0 && std::abs(ds[r]-amax/127.f)>2*std::numeric_limits<float>::epsilon()*std::max(1.f,amax/127.f)))throw std::runtime_error("dynamic scale mismatch got="+std::to_string(ds[r])+" expected="+std::to_string(amax/127.f));
      for(int64_t c=0;c<k;++c) {
        int ref=amax==0?0:int(std::nearbyint(fp32(x[r*k+c])/(amax/127.f)));
        double err=std::abs(int(q[r*k+c])-ref);max_quant_error=std::max(max_quant_error,err);
        if(err>1)throw std::runtime_error("dynamic quant mismatch");
      }
      for(int64_t col=0;col<n;++col) {
        int64_t sum=0;for(int64_t c=0;c<k;++c)sum+=int(q[r*k+c])*int(weight[col*k+c]);
        const float expected=fp32(bf16(float(sum)*ds[r]*scale[col]));const float actual=fp32(y[r*n+col]);
        double err=std::abs(double(expected)-actual);max_error=std::max(max_error,err);
        if(!std::isfinite(actual) || err>0.015625*std::abs(expected)+0.0001)throw std::runtime_error("W8A8 numerical mismatch");
        ++checked;
      }
    }
    return y;
  }
  void run() {
    std::vector<std::vector<uint16_t>> eager;
    for(int v=0;v<3;++v){auto x=input(v);launch();check(aclrtSynchronizeStream(stream),"eager fence");eager.push_back(verify(x));}
    input(0);launch();check(aclrtSynchronizeStream(stream),"warmup");
    check(aclmdlRICaptureBegin(stream,ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),"capture begin");
    try {launch();} catch(...) {aclmdlRICaptureEnd(stream,&graph);throw;}
    check(aclmdlRICaptureEnd(stream,&graph),"capture end");
    for(int pass=0;pass<3;++pass)for(int v=0;v<3;++v) {
      auto x=input(v);check(aclmdlRIExecuteAsync(graph,stream),"replay");check(aclrtSynchronizeStream(stream),"graph fence");
      if(verify(x)!=eager[v])throw std::runtime_error("graph/eager differ");
    }
  }
  ~Runner() {
    if(stream)aclrtSynchronizeStream(stream);
    if(graph)aclmdlRIDestroy(graph);
    if(dq)aclDestroyAclOpExecutor(dq);
    if(mm)aclDestroyAclOpExecutor(mm);
    for(auto &t:tensors){aclDestroyTensor(t.t);aclrtFree(t.data);}
    if(workspace)aclrtFree(workspace);
    if(stream && owns_stream)aclrtDestroyStream(stream);
  }
};
