// TP8 projection -> canonical all-gather -> EP8 SwiGLU experts -> weighted all-reduce.
// Qualification pipeline with explicit routing coefficients, not a full decoder.
#include "openpangu_probe_common.h"
#include <aclnnop/aclnn_silu.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_cast.h>
#include <hccl/hccl.h>
#include <hccl/hccl_comm.h>
#include <array>
#include <thread>
#include <cstdlib>
using Launch=aclnnStatus(*)(void*,uint64_t,aclOpExecutor*,aclrtStream);
struct Extra {aclOpExecutor *exec=nullptr;uint64_t bytes=0;Launch launch=nullptr;};
struct Evidence {std::vector<std::vector<uint16_t>> shard,canonical;std::vector<std::vector<float>> local,reduced;double max_error=0;};
struct Pipeline {
  Runner tp,gate,up,down;
  int rank;int64_t full_n;HcclComm comm=nullptr;aclmdlRI graph=nullptr;
  std::array<Extra,4> extra{};void *workspace=nullptr;uint64_t workspace_bytes=0;
  std::vector<uint16_t> input_values;
  Pipeline(const std::string &dir,int r,int batch):tp((dir+"/tp.bin").c_str(),batch),
    gate((dir+"/rank"+std::to_string(r)+"-gate.bin").c_str(),batch),
    up((dir+"/rank"+std::to_string(r)+"-up.bin").c_str(),batch),
    down((dir+"/rank"+std::to_string(r)+"-down.bin").c_str(),batch),rank(r),full_n(tp.n) {
    if(tp.n%8 || gate.k!=tp.n || up.k!=gate.k || up.n!=gate.n || down.k!=gate.n || down.n!=tp.n)throw std::runtime_error("pipeline fixture shapes");
    tp.n/=8;
    tp.weight=std::vector<int8_t>(tp.weight.begin()+r*tp.n*tp.k,tp.weight.begin()+(r+1)*tp.n*tp.k);
    tp.scale=std::vector<float>(tp.scale.begin()+r*tp.n,tp.scale.begin()+(r+1)*tp.n);
  }
  void query(size_t j) {
    auto &e=extra[j];
    if(j==0){check(aclnnSiluGetWorkspaceSize(gate.tensors[5].t,tp.tensors[8].t,&e.bytes,&e.exec),"silu prepare");e.launch=aclnnSilu;}
    if(j==1){check(aclnnMulGetWorkspaceSize(tp.tensors[8].t,up.tensors[5].t,down.tensors[0].t,&e.bytes,&e.exec),"gate multiply prepare");e.launch=aclnnMul;}
    if(j==2){check(aclnnCastGetWorkspaceSize(down.tensors[5].t,ACL_FLOAT,tp.tensors[9].t,&e.bytes,&e.exec),"cast prepare");e.launch=aclnnCast;}
    if(j==3){check(aclnnMulGetWorkspaceSize(tp.tensors[9].t,tp.tensors[10].t,tp.tensors[11].t,&e.bytes,&e.exec),"weight prepare");e.launch=aclnnMul;}
  }
  void prepare(const HcclRootInfo &root) {
    check(HcclCommInitRootInfo(8,&root,rank,&comm),"HCCL init");
    tp.prepare();gate.prepare(tp.stream);up.prepare(tp.stream);down.prepare(tp.stream);
    tp.tensor({8,tp.m,tp.n},ACL_BF16,2); // 6 rank-major gather
    tp.tensor({tp.m,full_n},ACL_BF16,2); // 7 canonical input
    tp.tensor({tp.m,gate.n},ACL_BF16,2); // 8 silu
    tp.tensor({tp.m,full_n},ACL_FLOAT,4); // 9 expert output
    tp.tensor({tp.m,1},ACL_FLOAT,4); // 10 routing weight
    tp.tensor({tp.m,full_n},ACL_FLOAT,4); // 11 local weighted contribution
    tp.tensor({tp.m,full_n},ACL_FLOAT,4); // 12 reduced output
    for(size_t j=0;j<extra.size();++j){query(j);workspace_bytes=std::max(workspace_bytes,extra[j].bytes);}
    check(aclrtMalloc(&workspace,std::max(uint64_t(64),workspace_bytes),ACL_MEM_MALLOC_HUGE_FIRST),"extra workspace");
  }
  void op(size_t j) {
    auto &e=extra[j];if(!e.exec)query(j);
    if(e.bytes>workspace_bytes)throw std::runtime_error("workspace grew");
    check(e.launch(workspace,e.bytes,e.exec,tp.stream),"ep operation");e.exec=nullptr;
  }
  void copy(void *dst,const void *src,size_t bytes){check(aclrtMemcpyAsync(dst,bytes,src,bytes,ACL_MEMCPY_DEVICE_TO_DEVICE,tp.stream),"pipeline copy");}
  void launch() {
    tp.launch();
    check(HcclAllGather(tp.tensors[5].data,tp.tensors[6].data,tp.m*tp.n,HCCL_DATA_TYPE_BFP16,comm,tp.stream),"TP gather");
    for(int peer=0;peer<8;++peer)for(int64_t row=0;row<tp.m;++row)
      copy(static_cast<uint16_t*>(tp.tensors[7].data)+row*full_n+peer*tp.n,static_cast<uint16_t*>(tp.tensors[6].data)+(peer*tp.m+row)*tp.n,tp.n*2);
    copy(gate.tensors[0].data,tp.tensors[7].data,tp.m*full_n*2);
    copy(up.tensors[0].data,tp.tensors[7].data,tp.m*full_n*2);
    gate.launch();up.launch();op(0);op(1);down.launch();op(2);op(3);
    check(HcclAllReduce(tp.tensors[11].data,tp.tensors[12].data,tp.m*full_n,HCCL_DATA_TYPE_FP32,HCCL_REDUCE_SUM,comm,tp.stream),"EP sum");
  }
  float coefficient(int variant,int64_t token) const {
    if(variant==0)return 1.f/8;
    if(variant==1)return rank==((token+3)%8)?1.f:0.f;
    return 0.f;
  }
  void input(int variant){input_values=tp.input(variant==2?0:variant);std::vector<float> weights(tp.m);for(int64_t t=0;t<tp.m;++t)weights[t]=coefficient(variant,t);tp.upload(10,weights.data());}
  void validate(int variant,Evidence &e) {
    tp.verify(input_values);
    auto canonical=tp.read<uint16_t>(7);auto g=gate.verify(canonical);auto u=up.verify(canonical);auto x=down.read<uint16_t>(0);
    for(size_t j=0;j<x.size();++j){float v=fp32(g[j]);float expected=fp32(bf16(fp32(bf16(v/(1+std::exp(-v))))*fp32(u[j])));float actual=fp32(x[j]);if(!std::isfinite(actual)||std::abs(actual-expected)>0.016*std::abs(expected)+0.0001)throw std::runtime_error("SwiGLU mismatch");}
    auto y=down.verify(x);auto local=tp.read<float>(11);
    for(int64_t t=0;t<tp.m;++t)for(int64_t j=0;j<full_n;++j){float expected=fp32(y[t*full_n+j])*coefficient(variant,t);if(local[t*full_n+j]!=expected)throw std::runtime_error("route weight mismatch");}
    e.shard.push_back(tp.read<uint16_t>(5));e.canonical.push_back(canonical);e.local.push_back(local);e.reduced.push_back(tp.read<float>(12));
    e.max_error=std::max({e.max_error,tp.max_error,gate.max_error,up.max_error,down.max_error});
  }
  void run(Evidence &e) {
    for(int v=0;v<3;++v){input(v);launch();check(aclrtSynchronizeStream(tp.stream),"eager fence");validate(v,e);}
    input(0);launch();check(aclrtSynchronizeStream(tp.stream),"warmup");
    check(aclmdlRICaptureBegin(tp.stream,ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),"capture begin");
    try{launch();}catch(...){aclmdlRICaptureEnd(tp.stream,&graph);throw;}
    check(aclmdlRICaptureEnd(tp.stream,&graph),"capture end");
    for(int pass=0;pass<3;++pass)for(int v=0;v<3;++v){input(v);check(aclmdlRIExecuteAsync(graph,tp.stream),"pipeline replay");check(aclrtSynchronizeStream(tp.stream),"replay fence");
      if(tp.read<float>(12)!=e.reduced[v] || tp.read<uint16_t>(7)!=e.canonical[v])throw std::runtime_error("pipeline graph/eager mismatch");}
  }
  ~Pipeline(){if(tp.stream)aclrtSynchronizeStream(tp.stream);if(graph)aclmdlRIDestroy(graph);for(auto &e:extra)if(e.exec)aclDestroyAclOpExecutor(e.exec);if(workspace)aclrtFree(workspace);if(comm)HcclCommDestroy(comm);}
};
int main(int argc,char **argv) {
  if(argc!=3){std::cerr<<"usage: openpangu_parallel_probe FIXTURE_DIR BATCH\n";return 2;}
  try {
    const int batch=std::stoi(argv[2]);if(batch!=1 && batch!=4)throw std::runtime_error("qualified batches are 1 and 4");
    const char *mode=std::getenv("HCCL_OP_EXPANSION_MODE");if(!mode || std::string(mode)!="HOST")throw std::runtime_error("HCCL_OP_EXPANSION_MODE=HOST required");
    check(aclInit(nullptr),"init");check(aclrtSetDevice(0),"bootstrap device");HcclRootInfo root;check(HcclGetRootInfo(&root),"root");
    std::array<Evidence,8> evidence;std::vector<std::thread> threads;
    for(int rank=0;rank<8;++rank)threads.emplace_back([&,rank]{try {check(aclrtSetDevice(rank),"device");{Pipeline p(argv[1],rank,batch);p.prepare(root);p.run(evidence[rank]);}check(aclrtResetDevice(rank),"reset");}catch(const std::exception &e){std::cerr<<"rank "<<rank<<": "<<e.what()<<'\n';std::_Exit(1);}});
    for(auto &t:threads)t.join();
    size_t values=0;double max_error=0;
    const size_t n=evidence[0].canonical[0].size()/batch,local_n=n/8;
    for(int v=0;v<3;++v)for(int rank=0;rank<8;++rank) {
      for(int row=0;row<batch;++row)for(int peer=0;peer<8;++peer)for(size_t j=0;j<local_n;++j)
        if(evidence[rank].canonical[v][row*n+peer*local_n+j]!=evidence[peer].shard[v][row*local_n+j])throw std::runtime_error("TP gather rank/token layout mismatch");
      for(size_t j=0;j<evidence[rank].reduced[v].size();++j){double sum=0;for(auto &e:evidence)sum+=e.local[v][j];double actual=evidence[rank].reduced[v][j];if(!std::isfinite(actual)||std::abs(actual-sum)>0.00002*std::abs(sum)+0.0001)throw std::runtime_error("EP reduction mismatch");++values;}
      max_error=std::max(max_error,evidence[rank].max_error);
    }
    check(aclFinalize(),"finalize");std::cout<<"{\"status\":\"passed\",\"tp\":8,\"ep\":8,\"batch\":"<<batch<<",\"experts_tested\":[0,32,64,96,128,160,192,224],\"routing_cases\":3,\"graph_passes\":3,\"reduced_values_checked\":"<<values<<",\"max_linear_abs_error\":"<<max_error<<",\"scope\":\"real-weight TP projection and selected EP SwiGLU experts; explicit routes, not full decoder or router qualification\"}\n";return 0;
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
