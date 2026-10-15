#include "openpangu_probe_common.h"
#include "../src/openpangu_ops.h"
#include <aclnn_ai_infra_manifold_constrained_hyper_connection_pre.h>
#include <aclnn_manifold_constrained_hyper_connection_sinkhorn_enhance.h>
#include <aclnn_ai_infra_manifold_constrained_hyper_connection_post.h>
#include <array>
using openpangu::DeviceTensor;using openpangu::FloatFixture;using openpangu::Operation;
struct Mhc {
  int batch;aclrtStream stream=nullptr;aclmdlRI graph=nullptr;
  FloatFixture w,a,b,g;
  DeviceTensor x,phi,alpha,bias,pre,post,res,rms,mix,pre_weights,sink,branch,output;
  std::array<Operation,3> ops;std::vector<float> folded;
  double max_f32_error=0,max_bf16_error=0;size_t checked_values=0;
  Mhc(const std::string &dir,int n):batch(n),w(dir+"/phi.bin"),a(dir+"/alpha.bin"),b(dir+"/beta.bin"),g(dir+"/gamma.bin"),
    x({n,4,2560},ACL_BF16,2),phi({24,10240},ACL_FLOAT,4),alpha({3},ACL_FLOAT,4),bias({24},ACL_FLOAT,4),
    pre({n,2560},ACL_BF16,2),post({n,4},ACL_FLOAT,4),res({n,4,4},ACL_FLOAT,4),rms({n},ACL_FLOAT,4),mix({n,24},ACL_FLOAT,4),pre_weights({n,4},ACL_FLOAT,4),sink({n,4,4},ACL_FLOAT,4),branch({n,2560},ACL_BF16,2),output({n,4,2560},ACL_BF16,2) {
    if(w.shape!=std::vector<int64_t>{24,10240}||a.shape!=std::vector<int64_t>{3}||b.shape!=std::vector<int64_t>{24}||g.shape!=std::vector<int64_t>{10240})throw std::runtime_error("mHC fixture shape");
    check(aclrtCreateStream(&stream),"mHC stream");folded=w.values;for(size_t j=0;j<folded.size();++j)folded[j]*=g.values[j%10240];phi.upload(folded);alpha.upload(a.values);bias.upload(b.values);
    for(int i=0;i<3;++i){query(i);ops[i].allocate();}
  }
  void query(int i){auto&o=ops[i];
    if(i==0)check(aclnnAiInfraManifoldConstrainedHyperConnectionPreGetWorkspaceSize(x.tensor,phi.tensor,alpha.tensor,bias.tensor,nullptr,1,1e-6,1e-6,pre.tensor,post.tensor,res.tensor,rms.tensor,mix.tensor,pre_weights.tensor,&o.bytes,&o.executor),"mHC pre prepare");
    if(i==1)check(aclnnManifoldConstrainedHyperConnectionSinkhornEnhanceGetWorkspaceSize(res.tensor,0,1e-6,20,sink.tensor,nullptr,nullptr,&o.bytes,&o.executor),"Sinkhorn prepare");
    if(i==2)check(aclnnAiInfraManifoldConstrainedHyperConnectionPostGetWorkspaceSize(x.tensor,sink.tensor,branch.tensor,post.tensor,output.tensor,&o.bytes,&o.executor),"mHC post prepare");
  }
  void launch(){std::array<openpangu::Launcher,3> f={aclnnAiInfraManifoldConstrainedHyperConnectionPre,aclnnManifoldConstrainedHyperConnectionSinkhornEnhance,aclnnAiInfraManifoldConstrainedHyperConnectionPost};for(int i=0;i<3;++i){if(!ops[i].executor)query(i);ops[i].run(f[i],stream);}}
  void input(int variant){std::vector<uint16_t> residual(batch*10240),layer(batch*2560);for(size_t j=0;j<residual.size();++j)residual[j]=bf16(variant==2?0.f:float(int((j*19+variant*31)%211)-105)/64.f);for(size_t j=0;j<layer.size();++j)layer[j]=bf16(float(int((j*7+variant*13)%97)-48)/64.f);x.upload(residual);branch.upload(layer);}
  void close(double actual,double expected,bool bf,const char *where){double error=std::abs(actual-expected);if(bf)max_bf16_error=std::max(max_bf16_error,error);else max_f32_error=std::max(max_f32_error,error);if(!std::isfinite(actual)||error>(bf?1e-4+.016*std::abs(expected):2e-5+2e-5*std::abs(expected)))throw std::runtime_error(std::string(where)+" actual="+std::to_string(actual)+" expected="+std::to_string(expected));++checked_values;}
  void verify(){auto hx=x.read<uint16_t>(),hl=branch.read<uint16_t>(),hin=pre.read<uint16_t>(),out=output.read<uint16_t>();auto hp=post.read<float>(),hr=res.read<float>(),hs=sink.read<float>(),ri=rms.read<float>(),hm=mix.read<float>(),hw=pre_weights.read<float>();
    for(int t=0;t<batch;++t){double square=0;for(int j=0;j<10240;++j){double v=fp32(hx[t*10240+j]);square+=v*v;}double inv=1/std::sqrt(square/10240+1e-6);close(ri[t],inv,false,"mHC inverse RMS");std::array<double,24> m{};std::array<double,4> before{},after{};std::array<double,16> residual{};
      for(int j=0;j<24;++j){double sum=0;for(int k=0;k<10240;++k)sum+=double(fp32(hx[t*10240+k]))*folded[j*10240+k];m[j]=sum*inv;close(hm[t*24+j],sum,false,"mHC projection");}
      for(int j=0;j<4;++j){before[j]=1/(1+std::exp(-(m[j]*a.values[0]+b.values[j])))+1e-6;after[j]=2/(1+std::exp(-(m[j+4]*a.values[1]+b.values[j+4])));close(hw[t*4+j],before[j],false,"mHC pre gate");close(hp[t*4+j],after[j],false,"mHC post gate");}
      for(int j=0;j<16;++j){residual[j]=m[j+8]*a.values[2]+b.values[j+8];close(hr[t*16+j],residual[j],false,"mHC residual logits");}
      for(int j=0;j<2560;++j){double sum=0;for(int i=0;i<4;++i)sum+=before[i]*fp32(hx[t*10240+i*2560+j]);close(fp32(hin[t*2560+j]),fp32(bf16(float(sum))),true,"mHC pre output");}
      for(int i=0;i<4;++i){double maximum=*std::max_element(residual.begin()+i*4,residual.begin()+(i+1)*4),sum=0;for(int j=0;j<4;++j){residual[i*4+j]=std::exp(residual[i*4+j]-maximum);sum+=residual[i*4+j];}for(int j=0;j<4;++j)residual[i*4+j]=residual[i*4+j]/sum+1e-6;}
      auto column=[&]{for(int j=0;j<4;++j){double sum=1e-6;for(int i=0;i<4;++i)sum+=residual[i*4+j];for(int i=0;i<4;++i)residual[i*4+j]/=sum;}};column();
      for(int pass=1;pass<20;++pass){for(int i=0;i<4;++i){double sum=1e-6;for(int j=0;j<4;++j)sum+=residual[i*4+j];for(int j=0;j<4;++j)residual[i*4+j]/=sum;}column();}
      for(int j=0;j<16;++j)close(hs[t*16+j],residual[j],false,"Sinkhorn output");
      for(int i=0;i<4;++i)for(int j=0;j<2560;++j){double sum=after[i]*fp32(hl[t*2560+j]);for(int k=0;k<4;++k)sum+=residual[k*4+i]*fp32(hx[t*10240+k*2560+j]);close(fp32(out[t*10240+i*2560+j]),fp32(bf16(float(sum))),true,"mHC post output");}
    }
  }
  void run(){std::vector<std::vector<uint16_t>> eager;for(int v=0;v<3;++v){input(v);launch();check(aclrtSynchronizeStream(stream),"mHC eager fence");verify();eager.push_back(output.read<uint16_t>());}check(aclmdlRICaptureBegin(stream,ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),"mHC capture");launch();check(aclmdlRICaptureEnd(stream,&graph),"mHC capture end");for(int pass=0;pass<3;++pass)for(int v=0;v<3;++v){input(v);check(aclmdlRIExecuteAsync(graph,stream),"mHC replay");check(aclrtSynchronizeStream(stream),"mHC replay fence");verify();if(output.read<uint16_t>()!=eager[v])throw std::runtime_error("mHC graph/eager mismatch");}}
  ~Mhc(){if(stream)aclrtSynchronizeStream(stream);if(graph)aclmdlRIDestroy(graph);if(stream)aclrtDestroyStream(stream);}
};
int main(int argc,char **argv){if(argc!=3){std::cerr<<"usage: openpangu_mhc_probe FIXTURES BATCH\n";return 2;}try{int b=std::stoi(argv[2]);if(b!=1&&b!=4)throw std::runtime_error("batch 1 or 4");check(aclInit(nullptr),"init");check(aclrtSetDevice(0),"device");{Mhc m(argv[1],b);m.run();std::cout<<"{\"status\":\"passed\",\"batch\":"<<b<<",\"cases\":3,\"graph_passes\":3,\"checked_values\":"<<m.checked_values<<",\"max_f32_abs_error\":"<<m.max_f32_error<<",\"max_bf16_abs_error\":"<<m.max_bf16_error<<"}\n";}check(aclrtResetDevice(0),"reset");check(aclFinalize(),"finalize");}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
