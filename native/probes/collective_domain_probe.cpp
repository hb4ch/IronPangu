// Offline communication contract: configurable uncaptured domain beside a HOST
// decode graph. Synthetic buffers only; no model, production fallback or reset.
#include <acl/acl.h>
#include <hccl/hccl.h>
#include <hccl/hccl_comm.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

static void check(int status,const char *what) {
  if(status)throw std::runtime_error(std::string(what)+": "+std::to_string(status));
}
static size_t number(const char *value,size_t maximum) {
  std::string text(value);size_t end=0;
  auto n=std::stoull(text,&end);
  if(text.empty() || text[0]=='-' || end!=text.size() || !n || n>maximum)
    throw std::runtime_error("invalid probe bound");
  return n;
}
struct Sample {double enqueue_us,complete_us;};
using Clock=std::chrono::steady_clock;
static double us(Clock::time_point begin,Clock::time_point end) {
  return std::chrono::duration<double,std::micro>(end-begin).count();
}
int main(int argc,char **argv) {
  try {
    if(argc!=5)throw std::runtime_error("usage: collective_domain_probe MODE(1=HOST,2=AI_CPU,3=AIV) RANKS ELEMENTS REPETITIONS");
    const auto mode=number(argv[1],3),ranks=number(argv[2],8),elements=number(argv[3],16*1024*1024),repeats=number(argv[4],1000);
    check(aclInit(nullptr),"acl init");
    check(aclrtSetDevice(0),"bootstrap device");
    HcclRootInfo bulk_root{},decode_root{};
    check(HcclGetRootInfo(&bulk_root),"bulk root");check(HcclGetRootInfo(&decode_root),"decode root");
    std::vector<std::vector<Sample>> samples(ranks);
    std::vector<std::thread> workers;
    for(size_t rank=0;rank<ranks;++rank)workers.emplace_back([&,rank] {
      try {
        check(aclrtSetDevice(rank),"device");aclrtStream stream=nullptr;
        check(aclrtCreateStream(&stream),"stream");
        HcclComm bulk=nullptr,decode=nullptr;HcclCommConfig config;
        HcclCommConfigInit(&config);config.hcclOpExpansionMode=mode;
        check(HcclCommInitRootInfoConfig(ranks,&bulk_root,rank,&config,&bulk),"bulk domain");
        HcclCommConfigInit(&config);config.hcclOpExpansionMode=1;
        check(HcclCommInitRootInfoConfig(ranks,&decode_root,rank,&config,&decode),"decode domain");
        const auto bytes=elements*sizeof(float);void *input=nullptr,*output=nullptr;
        check(aclrtMalloc(&input,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"input");
        check(aclrtMalloc(&output,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"output");
        std::vector<float> host(elements,float(rank+1));
        check(aclrtMemcpy(input,bytes,host.data(),bytes,ACL_MEMCPY_HOST_TO_DEVICE),"synthetic input");
        auto reduce=[&](HcclComm comm,size_t count) {
          check(HcclAllReduce(input,output,count,HCCL_DATA_TYPE_FP32,HCCL_REDUCE_SUM,comm,stream),"reduce");
        };
        auto verify=[&](size_t count) {
          check(aclrtMemcpy(host.data(),bytes,output,count*sizeof(float),ACL_MEMCPY_DEVICE_TO_HOST),"synthetic output");
          if(!std::all_of(host.begin(),host.begin()+count,[&](float x){return x==float(ranks*(ranks+1)/2);}))
            throw std::runtime_error("reduction value mismatch");
        };
        for(size_t i=0;i<3+repeats;++i) {
          auto begin=Clock::now();reduce(bulk,elements);auto submitted=Clock::now();
          check(aclrtSynchronizeStream(stream),"bulk completion");auto done=Clock::now();
          if(i>=3)samples[rank].push_back({us(begin,submitted),us(begin,done)});
        }
        verify(elements);
        // No graph capture of bulk prefill. Only the distinct HOST domain enters
        // this small decode-like graph, while the bulk communicator remains live.
        const auto small=std::min<size_t>(elements,2560);
        reduce(decode,small);check(aclrtSynchronizeStream(stream),"decode warmup");
        check(aclmdlRICaptureBegin(stream,ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),"capture begin");
        reduce(decode,small);aclmdlRI graph=nullptr;
        check(aclmdlRICaptureEnd(stream,&graph),"capture end");
        for(int i=0;i<3;++i) {
          check(aclmdlRIExecuteAsync(graph,stream),"replay");
          check(aclrtSynchronizeStream(stream),"replay completion");verify(small);
          reduce(bulk,elements);check(aclrtSynchronizeStream(stream),"interleaved completion");verify(elements);
        }
        check(aclmdlRIDestroy(graph),"graph destroy");
        check(HcclCommDestroy(decode),"decode destroy");check(HcclCommDestroy(bulk),"bulk destroy");
        check(aclrtFree(output),"free output");check(aclrtFree(input),"free input");
        check(aclrtDestroyStream(stream),"stream destroy");
      } catch(const std::exception &e) {
        // Fail the entire diagnostic process rather than strand peers waiting in
        // a collective after one worker fails. Never reset shared hardware.
        std::cerr<<"rank "<<rank<<": "<<e.what()<<std::endl;std::_Exit(2);
      }
    });
    for(auto &worker:workers)worker.join();
    check(aclFinalize(),"finalize");
    std::cout<<"{\"passed\":true,\"requested_bulk_mode\":"<<mode<<",\"decode_mode\":1,\"ranks\":"<<ranks
      <<",\"elements\":"<<elements<<",\"input_bytes_per_rank\":"<<elements*4
      <<",\"graph_replays_per_rank\":3,\"samples\":[";
    for(size_t rank=0;rank<ranks;++rank)for(size_t i=0;i<samples[rank].size();++i) {
      if(rank||i)std::cout<<",";
      const auto &s=samples[rank][i];
      std::cout<<"{\"rank\":"<<rank<<",\"iteration\":"<<i<<",\"enqueue_us\":"<<s.enqueue_us<<",\"complete_us\":"<<s.complete_us<<"}";
    }
    std::cout<<"]}"<<std::endl;return 0;
  } catch(const std::exception &e) {std::cerr<<e.what()<<std::endl;return 1;}
}
