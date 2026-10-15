#include "openpangu_probe_common.h"
#include "../src/openpangu_ops.h"
#include <aclnn_ai_infra_fused_causal_conv1d.h>
#include <array>
using openpangu::DeviceTensor;using openpangu::FloatFixture;using openpangu::Operation;
struct Mome {
  int dim;FloatFixture fixture;aclrtStream stream=nullptr;aclmdlRI graph=nullptr;
  DeviceTensor x,weight,state,starts,slots,computed,out;Operation op;
  std::vector<uint16_t> weights,reference_state;std::array<int,4> counts{};
  size_t checked_values=0;double max_error=0;
  explicit Mome(const std::string &path):dim(FloatFixture(path).shape.at(0)),fixture(path),x({4,dim},ACL_BF16,2),weight({3,dim},ACL_BF16,2),state({4,2,dim},ACL_BF16,2),starts({3},ACL_INT32,4),slots({2},ACL_INT32,4),computed({2},ACL_INT32,4),out({4,dim},ACL_BF16,2){
    if(fixture.shape!=std::vector<int64_t>{dim,1,3} || (dim!=512&&dim!=1024&&dim!=6144))throw std::runtime_error("MoME checkpoint shape");
    weights.resize(dim*3);for(int j=0;j<dim;++j)for(int k=0;k<3;++k)weights[k*dim+j]=bf16(fixture.values[j*3+k]);weight.upload(weights);
    check(aclrtCreateStream(&stream),"MoME stream");query();op.allocate();reset();
  }
  void query(){check(aclnnAiInfraFusedCausalConv1dGetWorkspaceSize(x.tensor,weight.tensor,state.tensor,starts.tensor,slots.tensor,nullptr,nullptr,nullptr,computed.tensor,nullptr,nullptr,nullptr,0,-1,0,4,1,128,1,true,out.tensor,&op.bytes,&op.executor),"MoME prepare");}
  void launch(){if(!op.executor)query();op.run(aclnnAiInfraFusedCausalConv1d,stream);}
  void reset(){reference_state.assign(8*dim,bf16(7.25f));state.upload(reference_state);counts.fill(0);}
  std::vector<uint16_t> step(int v){std::vector<int32_t> ids={2,0},offset={0,3,4};
    if(v==1){ids={0,2};offset={0,3,4};}
    if(v==2){offset={0,2,4};}
    if(v==3){counts[2]=0;offset={0,1,4};}
    if(v==4){ids={2,-1};offset={0,2,4};}
    if(v==5){ids={0,2};offset={0,4,4};}
    std::vector<int32_t> n={counts[ids[0]],ids[1]<0?0:counts[ids[1]]};std::vector<uint16_t> input(4*dim),expected(4*dim);
    for(int lane=0;lane<2;++lane){int length=offset[lane+1]-offset[lane];for(int t=0;t<length;++t)for(int j=0;j<dim;++j)input[(offset[lane]+t)*dim+j]=bf16(float(((j*13+(t+n[lane])*17+lane*7+v*11)%101)-50)/64.f);}
    x.upload(input);starts.upload(offset);slots.upload(ids);computed.upload(n);
    for(int lane=0;lane<2;++lane){int slot=ids[lane],length=offset[lane+1]-offset[lane];if(slot<0){for(int t=offset[lane];t<offset[lane+1];++t)for(int j=0;j<dim;++j)expected[t*dim+j]=input[t*dim+j];continue;}if(!length)continue;
      if(n[lane]==0)std::fill(reference_state.begin()+slot*2*dim,reference_state.begin()+(slot+1)*2*dim,uint16_t(0));
      for(int t=offset[lane];t<offset[lane+1];++t){for(int j=0;j<dim;++j){float cur=fp32(input[t*dim+j]),sum=0;if(counts[slot]>=2)sum=fp32(reference_state[slot*2*dim+j])*fp32(weights[j])+fp32(reference_state[slot*2*dim+dim+j])*fp32(weights[dim+j])+cur*fp32(weights[2*dim+j]);expected[t*dim+j]=bf16(cur+sum);reference_state[slot*2*dim+j]=reference_state[slot*2*dim+dim+j];reference_state[slot*2*dim+dim+j]=input[t*dim+j];}++counts[slot];}
    }
    return expected;
  }
  void verify(const std::vector<uint16_t>& expected){auto actual=x.read<uint16_t>();for(size_t j=0;j<actual.size();++j){float ref=fp32(expected[j]),v=fp32(actual[j]);double error=std::abs(v-ref);max_error=std::max(max_error,error);if(!std::isfinite(v)||error>1e-4+.016*std::abs(ref))throw std::runtime_error("MoME output reference index="+std::to_string(j)+" actual="+std::to_string(v)+" expected="+std::to_string(ref));++checked_values;}if(state.read<uint16_t>()!=reference_state)throw std::runtime_error("MoME state/reset/padding mismatch");}
  void run(){std::vector<std::vector<uint16_t>> eager;reset();for(int v=0;v<6;++v){auto expected=step(v);launch();check(aclrtSynchronizeStream(stream),"MoME eager fence");verify(expected);eager.push_back(x.read<uint16_t>());}step(0);check(aclmdlRICaptureBegin(stream,ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),"MoME capture");launch();check(aclmdlRICaptureEnd(stream,&graph),"MoME capture end");for(int pass=0;pass<3;++pass){reset();for(int v=0;v<6;++v){auto expected=step(v);check(aclmdlRIExecuteAsync(graph,stream),"MoME replay");check(aclrtSynchronizeStream(stream),"MoME replay fence");verify(expected);if(x.read<uint16_t>()!=eager[v])throw std::runtime_error("MoME graph/eager mismatch");}}}
  ~Mome(){if(stream)aclrtSynchronizeStream(stream);if(graph)aclmdlRIDestroy(graph);if(stream)aclrtDestroyStream(stream);}
};
int main(int argc,char **argv){if(argc!=2){std::cerr<<"usage: openpangu_mome_probe WEIGHT_FIXTURE\n";return 2;}try{check(aclInit(nullptr),"init");check(aclrtSetDevice(0),"device");{Mome m(argv[1]);m.run();std::cout<<"{\"status\":\"passed\",\"channels\":"<<m.dim<<",\"steps\":6,\"graph_passes\":3,\"checked_values\":"<<m.checked_values<<",\"max_abs_error\":"<<m.max_error<<"}\n";}check(aclrtResetDevice(0),"reset");check(aclFinalize(),"finalize");}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
