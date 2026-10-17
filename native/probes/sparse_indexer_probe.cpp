#include "openpangu_probe_common.h"
#include "../src/openpangu_ops.h"
#include <aclnn_lightning_indexer_enhance.h>
#include <numeric>
#include <set>

// Independent exact-input score oracle. This qualifies the vendor boundary,
// not model projections or end-to-end long-context output.
struct IndexerProbe {
  static constexpr int capacity=65536, page=64, heads=24, dim=128, topk=2048, rows=2;
  openpangu::DeviceTensor q{{rows,heads,dim},ACL_BF16,2},
    k{{capacity/page,page,1,dim},ACL_BF16,2}, weights{{rows,heads},ACL_BF16,2},
    ends{{1},ACL_INT32,4}, lengths{{1},ACL_INT32,4}, table{{1,capacity/page},ACL_INT32,4},
    indices{{rows,1,topk},ACL_INT32,4}, values{{rows,1,topk},ACL_BF16,2};
  openpangu::Operation op;
  aclrtStream stream=nullptr;
  aclmdlRI graph=nullptr;
  size_t checked=0;
  std::vector<int32_t> blocks;
  IndexerProbe() {
    check(aclrtCreateStream(&stream),"indexer stream");
    std::vector<uint16_t> queries(rows*heads*dim), keys(capacity*dim), ws(rows*heads);
    for(int r=0;r<rows;++r) {
      queries[r*heads*dim]=bf16(r? -1.f:1.f);
      queries[r*heads*dim+1]=bf16((r?-1.f:1.f)/256);
      ws[r*heads]=bf16(1.f);
      queries[(r*heads+1)*dim]=bf16(r?1.f:-1.f);
      queries[(r*heads+1)*dim+1]=bf16((r?1.f:-1.f)/256);
      ws[r*heads+1]=bf16(.5f);
    }
    blocks.resize(capacity/page);
    for(int b=0;b<capacity/page;++b) blocks[b]=(b*37+19)%(capacity/page);
    for(int i=0;i<capacity;++i) {
      int code=(i*31337+12345)%capacity;
      int physical=blocks[i/page]*page+i%page;
      keys[physical*dim]=bf16(float(code/256-128)/256);
      keys[physical*dim+1]=bf16(float(code%256)/256);
    }
    q.upload(queries);k.upload(keys);weights.upload(ws);table.upload(blocks);
    ends.upload(std::vector<int32_t>{rows});set_length(capacity);
    prepare();op.allocate();
  }
  void set_length(int n){lengths.upload(std::vector<int32_t>{n});}
  void prepare() {
    check(aclnnLightningIndexerEnhanceGetWorkspaceSize(q.tensor,k.tensor,weights.tensor,
      ends.tensor,lengths.tensor,table.tensor,"TND","PA_BSND",topk,3,
      INT64_MAX,INT64_MAX,false,1,0,indices.tensor,values.tensor,&op.bytes,&op.executor),
      "indexer prepare");
  }
  void launch(){if(!op.executor)prepare();op.run(aclnnLightningIndexerEnhance,stream);}
  static float score(int token,int row) {
    int code=(token*31337+12345)%capacity;
    float dot=float(code-32768)/65536;
    float signed_dot=row?-dot:dot;
    return std::max(0.f,signed_dot)+.5f*std::max(0.f,-signed_dot);
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
        if(j>=count){if(id!=-1)throw std::runtime_error("indexer padding");continue;}
        if(id<0||id>=limit||!seen.insert(id).second||score(id,r)+1e-6f<cutoff)
          throw std::runtime_error("indexer causal/topk oracle mismatch length="+std::to_string(length)+
            " row="+std::to_string(r)+" id="+std::to_string(id)+" cutoff="+std::to_string(cutoff));
        ++checked;
      }
    }
  }
  void run() {
    const std::vector<int> cases{2,511,512,513,2048,2049,4096,16384,65536};
    for(int n:cases){set_length(n);launch();check(aclrtSynchronizeStream(stream),"indexer result fence");verify(n);}
    set_length(capacity);
    check(aclmdlRICaptureBegin(stream,ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),"indexer capture");
    launch();check(aclmdlRICaptureEnd(stream,&graph),"indexer capture end");
    for(int pass=0;pass<2;++pass)for(int n:cases){set_length(n);
      check(aclmdlRIExecuteAsync(graph,stream),"indexer replay");
      check(aclrtSynchronizeStream(stream),"indexer result fence");verify(n);}
  }
  ~IndexerProbe(){if(graph)aclmdlRIDestroy(graph);if(stream)aclrtDestroyStream(stream);}
};
int main(){try{check(aclInit(nullptr),"init");check(aclrtSetDevice(0),"device");
  {IndexerProbe p;p.run();std::cout<<"{\"status\":\"passed\",\"capacity\":65536,\"topk\":2048,\"checked_indices\":"
    <<p.checked<<",\"workspace_bytes\":"<<p.op.bytes<<",\"graph_replays\":18}\n";}
  check(aclrtResetDevice(0),"reset");check(aclFinalize(),"finalize");
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
