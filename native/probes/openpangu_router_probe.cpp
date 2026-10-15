#include "../src/openpangu_ops.h"
#include <iostream>
#include <numeric>
using namespace openpangu;
struct Evidence {double projection_error=0,score_error=0;size_t routes=0;};
static double sigmoid(double x){return x>=0?1/(1+std::exp(-x)):std::exp(x)/(1+std::exp(x));}
static void verify(Router &r,const std::vector<float> &bias,Evidence &e) {
  auto logits=r.logits.read<float>();auto scores=r.scores.read<float>();auto ids=r.ids.read<int32_t>();
  for(int64_t b=0;b<r.batch;++b) {
    std::vector<double> raw(r.experts),corrected(r.experts);std::vector<bool> chosen(r.experts,false);double sum=0,minimum=1e30;
    for(int64_t j=0;j<r.experts;++j){raw[j]=sigmoid(logits[b*r.experts+j]);corrected[j]=raw[j]+bias[j];}
    for(int64_t j=0;j<r.topk;++j){int id=ids[b*r.topk+j];if(id<0 || id>=r.experts || chosen[id])throw std::runtime_error("invalid/duplicate expert id");chosen[id]=true;sum+=raw[id];minimum=std::min(minimum,corrected[id]);}
    for(int64_t j=0;j<r.experts;++j)if(!chosen[j] && corrected[j]>minimum+2e-6)throw std::runtime_error("incorrect top-k selection");
    for(int64_t j=0;j<r.topk;++j){double expected=raw[ids[b*r.topk+j]]/(sum+1e-20)*2.5;double actual=scores[b*r.topk+j];double error=std::abs(actual-expected);e.score_error=std::max(e.score_error,error);if(!std::isfinite(actual) || error>2e-6+2e-5*std::abs(expected))throw std::runtime_error("routing weight differs from unbiased sigmoid reference");++e.routes;}
  }
}
struct Session {
  aclrtStream stream=nullptr;
  Session(){checked(aclrtCreateStream(&stream),"stream");}
  ~Session(){if(stream)aclrtSynchronizeStream(stream);if(stream)aclrtDestroyStream(stream);}
};
struct Graph {aclmdlRI graph=nullptr;aclrtStream stream; explicit Graph(aclrtStream s):stream(s){} ~Graph(){aclrtSynchronizeStream(stream);if(graph)aclmdlRIDestroy(graph);}};
static void run(const std::string &dir,int batch,Evidence &e) {
  Session s;FloatFixture w(dir+"/router-weight.bin"),bias(dir+"/router-bias.bin");
  // Graph handles must die before the tensors/workspaces they reference.
  Router r(batch,2560,256,8,s.stream);r.load(w,bias);Graph g(s.stream);
  std::vector<std::vector<float>> inputs,expected_scores;std::vector<std::vector<int32_t>> expected_ids;
  for(int v=0;v<3;++v){std::vector<float>x(batch*2560);for(size_t j=0;j<x.size();++j)x[j]=v==2?0.f:float(int((j*17+v*31)%101)-50)/64.f;inputs.push_back(x);r.input.upload(x);r.launch();checked(aclrtSynchronizeStream(s.stream),"router eager");
    auto y=r.logits.read<float>();for(int b=0;b<batch;++b)for(int n=0;n<256;++n){double ref=0;for(int k=0;k<2560;++k)ref+=double(x[b*2560+k])*w.values[n*2560+k];double error=std::abs(y[b*256+n]-ref);e.projection_error=std::max(e.projection_error,error);if(!std::isfinite(y[b*256+n]) || error>2e-5*(1+std::abs(ref)))throw std::runtime_error("FP32 router projection mismatch");}
    verify(r,bias.values,e);expected_scores.push_back(r.scores.read<float>());expected_ids.push_back(r.ids.read<int32_t>());
  }
  checked(aclmdlRICaptureBegin(s.stream,ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),"router capture");r.launch();checked(aclmdlRICaptureEnd(s.stream,&g.graph),"router capture end");
  for(int pass=0;pass<3;++pass)for(int v=0;v<3;++v){r.input.upload(inputs[v]);checked(aclmdlRIExecuteAsync(g.graph,s.stream),"router replay");checked(aclrtSynchronizeStream(s.stream),"router fence");if(r.ids.read<int32_t>()!=expected_ids[v] || r.scores.read<float>()!=expected_scores[v])throw std::runtime_error("router graph/eager mismatch");verify(r,bias.values,e);}
  checked(aclmdlRIDestroy(g.graph),"router graph destroy");g.graph=nullptr;
  // Isolate nonlinear selection: ties, bias-driven selections, and saturation.
  std::vector<std::vector<float>> logits_cases,bias_cases;expected_scores.clear();expected_ids.clear();
  for(int v=0;v<4;++v){std::vector<float> logits(batch*256,0),b(256,0);
    for(int j=0;j<256;++j){b[j]=v==1?float(255-j)/128.f:0;for(int t=0;t<batch;++t)logits[t*256+j]=v==0?0.f:v==1?float(j)/256.f:v==2?(j<8?80.f:-80.f):-80.f;}
    logits_cases.push_back(logits);bias_cases.push_back(b);r.logits.upload(logits);r.bias.upload(b);r.select();checked(aclrtSynchronizeStream(s.stream),"direct topk");verify(r,b,e);expected_scores.push_back(r.scores.read<float>());expected_ids.push_back(r.ids.read<int32_t>());
  }
  checked(aclmdlRICaptureBegin(s.stream,ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),"selection capture");r.select();checked(aclmdlRICaptureEnd(s.stream,&g.graph),"selection capture end");
  for(int pass=0;pass<3;++pass)for(int v=0;v<4;++v){r.logits.upload(logits_cases[v]);r.bias.upload(bias_cases[v]);checked(aclmdlRIExecuteAsync(g.graph,s.stream),"selection replay");checked(aclrtSynchronizeStream(s.stream),"selection fence");if(r.ids.read<int32_t>()!=expected_ids[v] || r.scores.read<float>()!=expected_scores[v])throw std::runtime_error("selection graph/eager mismatch");verify(r,bias_cases[v],e);}
  checked(aclmdlRIDestroy(g.graph),"selection graph destroy");g.graph=nullptr;
}
int main(int argc,char **argv){if(argc!=3){std::cerr<<"usage: openpangu_router_probe FIXTURES BATCH\n";return 2;}try{int b=std::stoi(argv[2]);if(b!=1 && b!=4)throw std::runtime_error("batch must be 1 or 4");checked(aclInit(nullptr),"init");checked(aclrtSetDevice(0),"device");Evidence e;run(argv[1],b,e);checked(aclrtResetDevice(0),"reset");checked(aclFinalize(),"finalize");std::cout<<"{\"status\":\"passed\",\"batch\":"<<b<<",\"real_weight_cases\":3,\"selection_edge_cases\":4,\"graph_passes\":3,\"routes_checked\":"<<e.routes<<",\"max_projection_abs_error\":"<<e.projection_error<<",\"max_score_abs_error\":"<<e.score_error<<"}\n";}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
