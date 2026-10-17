#include "openpangu_probe_common.h"
#include "../src/openpangu_ops.h"
#include <aclnn_ai_infra_quant_lightning_indexer.h>
#include <set>

// Independent oracle for the INT8/FP16 vendor boundary, including causal
// selection over shuffled pages. This does not qualify model quantization.
struct QuantIndexerProbe {
  static constexpr int capacity=65536, page=64, heads=24, dim=128, topk=2048, rows=2;
  openpangu::DeviceTensor q{{rows,heads,dim},ACL_INT8,1},
    k{{capacity/page,page,1,dim},ACL_INT8,1}, weights{{rows,heads},ACL_FLOAT16,2},
    qs{{rows,heads},ACL_FLOAT16,2}, ks{{capacity/page,page,1},ACL_FLOAT16,2},
    ends{{1},ACL_INT32,4}, lengths{{1},ACL_INT32,4}, table{{1,capacity/page},ACL_INT32,4},
    indices{{rows,1,topk},ACL_INT32,4};
  openpangu::Operation op;
  aclrtStream stream=nullptr;
  aclmdlRI graph=nullptr;
  size_t checked=0;
  std::vector<int32_t> blocks;
  static int code(int token){return (token*31337+12345)%capacity;}
  QuantIndexerProbe() {
    check(aclrtCreateStream(&stream),"quant indexer stream");
    std::vector<int8_t> queries(rows*heads*dim), keys(capacity*dim);
    std::vector<uint16_t> ws(rows*heads), query_scales(rows*heads,0x3c00), key_scales(capacity);
    for(int r=0;r<rows;++r) {
      queries[r*heads*dim]=r?-1:1;
      queries[(r*heads+1)*dim]=r?1:-1;
      ws[r*heads]=0x3c00; // FP16 1
      ws[r*heads+1]=0x3400; // FP16 .25
      query_scales[r*heads]=0x3800;
    }
    blocks.resize(capacity/page);
    for(int b=0;b<capacity/page;++b)blocks[b]=(b*37+19)%(capacity/page);
    for(int i=0;i<capacity;++i) {
      int c=code(i), physical=blocks[i/page]*page+i%page;
      keys[physical*dim]=c%255-127;
      // Exact powers of two: FP16 .125, .25, .5, 1. Varying scales
      // make a kernel that ignores or misaddresses scales fail selection.
      key_scales[physical]=uint16_t(0x3000+((c/255)%4)*0x400);
    }
    q.upload(queries);k.upload(keys);weights.upload(ws);qs.upload(query_scales);ks.upload(key_scales);
    table.upload(blocks);ends.upload(std::vector<int32_t>{rows});set_length(capacity);
    prepare();op.allocate();
  }
  void set_length(int n){lengths.upload(std::vector<int32_t>{n});}
  void prepare() {
    check(aclnnAiInfraQuantLightningIndexerGetWorkspaceSize(q.tensor,k.tensor,weights.tensor,
      qs.tensor,ks.tensor,ends.tensor,lengths.tensor,table.tensor,0,0,"TND","PA_BSND",
      topk,3,INT64_MAX,INT64_MAX,indices.tensor,&op.bytes,&op.executor),"quant indexer prepare");
  }
  void launch(){if(!op.executor)prepare();op.run(aclnnAiInfraQuantLightningIndexer,stream);}
  static float score(int token,int row) {
    int c=code(token);
    float dot=float(c%255-127)*std::ldexp(1.f,(c/255)%4-3)*(row?-1.f:1.f);
    return std::max(0.f,.5f*dot)+.25f*std::max(0.f,-dot);
  }
  void verify(int length) {
    auto ids=indices.read<int32_t>();
    for(int r=0;r<rows;++r) {
      int limit=length-rows+r+1, count=std::min(topk,limit);
      std::vector<float> scores(limit);
      for(int i=0;i<limit;++i)scores[i]=score(i,r);
      std::nth_element(scores.begin(),scores.begin()+count-1,scores.end(),std::greater<float>());
      float cutoff=scores[count-1];std::set<int> seen;
      for(int j=0;j<topk;++j) {
        int id=ids[r*topk+j];
        if(j>=count){if(id!=-1)throw std::runtime_error("quant indexer padding");continue;}
        if(id<0||id>=limit||!seen.insert(id).second||score(id,r)<cutoff)
          throw std::runtime_error("quant indexer causal/topk mismatch length="+std::to_string(length)+
            " row="+std::to_string(r)+" id="+std::to_string(id));
        ++checked;
      }
    }
  }
  void run() {
    const std::vector<int> cases{2,511,512,513,2047,2048,2049,4096,8192,16384,32768,65536};
    for(int n:cases){set_length(n);launch();check(aclrtSynchronizeStream(stream),"result fence");verify(n);}
    set_length(capacity);
    check(aclmdlRICaptureBegin(stream,ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),"capture begin");
    launch();check(aclmdlRICaptureEnd(stream,&graph),"capture end");
    for(int pass=0;pass<2;++pass)for(int n:cases){set_length(n);
      check(aclmdlRIExecuteAsync(graph,stream),"replay");
      check(aclrtSynchronizeStream(stream),"result fence");verify(n);}
  }
  ~QuantIndexerProbe(){if(graph)aclmdlRIDestroy(graph);if(stream)aclrtDestroyStream(stream);}
};
int main(){try{check(aclInit(nullptr),"init");check(aclrtSetDevice(0),"device");
  {QuantIndexerProbe p;p.run();std::cout<<"{\"status\":\"passed\",\"capacity\":65536,\"topk\":2048,\"checked_indices\":"
    <<p.checked<<",\"workspace_bytes\":"<<p.op.bytes<<",\"graph_replays\":24}\n";}
  check(aclrtResetDevice(0),"reset");check(aclFinalize(),"finalize");return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<"\n";return 1;}}
