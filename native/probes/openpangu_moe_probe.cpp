// Complete 256-expert layer qualification. Dense masked execution establishes
// correctness; grouped sparse dispatch and production scheduling remain separate.
#include "openpangu_probe_common.h"
#include "../src/openpangu_ops.h"
#include <aclnnop/aclnn_dequant_swiglu_quant.h>
#include <aclnnop/aclnn_swi_glu.h>
#include <aclnnop/aclnn_scatter.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_add.h>
#include <hccl/hccl.h>
#include <hccl/hccl_comm.h>
#include <array>
#include <functional>
#include <thread>
using openpangu::DeviceTensor;using openpangu::FloatFixture;using openpangu::Operation;
struct Op {
  Operation op;openpangu::Launcher launch;std::function<void()> query;
  Op(openpangu::Launcher l,std::function<void(Operation&)> q):launch(l){query=[this,q]{q(op);};query();op.allocate();}
  void run(aclrtStream s){if(!op.executor)query();op.run(launch,s);}
};
static void copy(void *d,const void *s,size_t n,aclrtStream stream){check(aclrtMemcpyAsync(d,n,s,n,ACL_MEMCPY_DEVICE_TO_DEVICE,stream),"device copy");}
struct FloatLinear {
  int64_t batch,n,k;std::vector<float> w;aclrtStream stream;
  DeviceTensor input,weight,output;std::unique_ptr<Op> mm;
  FloatLinear(const FloatFixture &f,int b,int rank,aclrtStream s):batch(b),n(f.shape.at(0)/8),k(f.shape.at(1)),stream(s),input({b,k},ACL_BF16,2),weight({k,n},ACL_BF16,2),output({b,n},ACL_BF16,2){
    if(f.shape.size()!=2 || f.shape[0]%8)throw std::runtime_error("shared TP shape");
    w.assign(f.values.begin()+rank*n*k,f.values.begin()+(rank+1)*n*k);std::vector<uint16_t> t(n*k);
    for(int64_t j=0;j<n;++j)for(int64_t c=0;c<k;++c)t[c*n+j]=bf16(w[j*k+c]);
    weight.upload(t);
    mm=std::make_unique<Op>(aclnnMatmul,[this](Operation&o){check(aclnnMatmulGetWorkspaceSize(input.tensor,weight.tensor,output.tensor,0,&o.bytes,&o.executor),"shared matmul");});
  }
  void run(){mm->run(stream);}
  void verify(){auto x=input.read<uint16_t>(),y=output.read<uint16_t>();for(int64_t b=0;b<batch;++b)for(int64_t j=0;j<n;++j){double sum=0;for(int64_t c=0;c<k;++c)sum+=double(fp32(x[b*k+c]))*w[j*k+c];float ref=fp32(bf16(float(sum))),actual=fp32(y[b*n+j]);if(!std::isfinite(actual)||std::abs(actual-ref)>1e-4+.016*std::abs(ref))throw std::runtime_error("shared matmul reference");}}
};
struct Expert {
  Runner gate,up,down;DeviceTensor packed,casted,weighted;aclTensor *coefficient=nullptr;
  std::unique_ptr<Op> swiglu,cast,mul,add;aclrtStream stream;
  Expert(const std::string &dir,int expert,int batch,aclrtStream s,DeviceTensor &coeff,DeviceTensor &sum,aclScalar *one):
    gate((dir+"/experts."+std::to_string(expert)+".gate_proj.bin").c_str(),batch),
    up((dir+"/experts."+std::to_string(expert)+".up_proj.bin").c_str(),batch),
    down((dir+"/experts."+std::to_string(expert)+".down_proj.bin").c_str(),batch),
    packed({batch,2048},ACL_BF16,2),casted({batch,2560},ACL_FLOAT,4),weighted({batch,2560},ACL_FLOAT,4),stream(s){
    if(gate.n!=1024||gate.k!=2560||up.n!=1024||up.k!=2560||down.n!=2560||down.k!=1024)throw std::runtime_error("expert shape");
    gate.prepare(s);up.prepare(s);down.prepare(s);
    int64_t shape[2]={batch,1},strides[2]={256,1},storage[2]={batch,256};
    coefficient=aclCreateTensor(shape,2,ACL_FLOAT,strides,expert,ACL_FORMAT_ND,storage,2,coeff.data);
    if(!coefficient)throw std::runtime_error("expert coefficient view");
    swiglu=std::make_unique<Op>(aclnnDequantSwigluQuant,[this](Operation&o){char mode[]="dynamic";check(aclnnDequantSwigluQuantGetWorkspaceSize(packed.tensor,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,true,mode,down.tensors[1].t,down.tensors[2].t,&o.bytes,&o.executor),"fused SwiGLU quant prepare");});
    cast=std::make_unique<Op>(aclnnCast,[this](Operation&o){check(aclnnCastGetWorkspaceSize(down.tensors[5].t,ACL_FLOAT,casted.tensor,&o.bytes,&o.executor),"expert cast");});
    mul=std::make_unique<Op>(aclnnMul,[this](Operation&o){check(aclnnMulGetWorkspaceSize(casted.tensor,coefficient,weighted.tensor,&o.bytes,&o.executor),"expert weight");});
    add=std::make_unique<Op>(aclnnInplaceAdd,[this,&sum,one](Operation&o){check(aclnnInplaceAddGetWorkspaceSize(sum.tensor,weighted.tensor,one,&o.bytes,&o.executor),"expert accumulate");});
  }
  void run(const void *input){copy(gate.tensors[0].data,input,gate.m*2560*2,stream);copy(up.tensors[0].data,input,gate.m*2560*2,stream);gate.launch();up.launch();
    for(int64_t b=0;b<gate.m;++b){copy(static_cast<uint16_t*>(packed.data)+b*2048,static_cast<uint16_t*>(gate.tensors[5].data)+b*1024,2048,stream);copy(static_cast<uint16_t*>(packed.data)+b*2048+1024,static_cast<uint16_t*>(up.tensors[5].data)+b*1024,2048,stream);}
    swiglu->run(stream);down.launch_quantized();cast->run(stream);mul->run(stream);add->run(stream);
  }
  void verify(const std::vector<uint16_t>& input){auto g=gate.verify(input),u=up.verify(input);auto q=down.read<int8_t>(1);auto scale=down.read<float>(2);auto y=down.read<uint16_t>(5);
    for(int64_t b=0;b<gate.m;++b){std::vector<float> activated(1024);float amax=0;
      for(int c=0;c<1024;++c){float v=fp32(g[b*1024+c]);activated[c]=v/(1+std::exp(-v))*fp32(u[b*1024+c]);amax=std::max(amax,std::abs(activated[c]));}
      if(!std::isfinite(scale[b])||scale[b]<0||std::abs(scale[b]-amax/127.f)>1e-6+2e-5*amax/127.f)throw std::runtime_error("fused SwiGLU scale");
      for(int c=0;c<1024;++c){int ref=amax==0?0:int(std::nearbyint(activated[c]*127.f/amax));if(std::abs(int(q[b*1024+c])-ref)>1)throw std::runtime_error("fused SwiGLU quantization");}
      for(int j=0;j<2560;++j){int64_t sum=0;for(int c=0;c<1024;++c)sum+=int(q[b*1024+c])*int(down.weight[j*1024+c]);float ref=fp32(bf16(float(sum)*scale[b]*down.scale[j])),actual=fp32(y[b*2560+j]);if(!std::isfinite(actual)||std::abs(actual-ref)>1e-4+.016*std::abs(ref))throw std::runtime_error("expert down reference");}
    }
  }
  ~Expert(){add.reset();mul.reset();cast.reset();swiglu.reset();if(coefficient)aclDestroyTensor(coefficient);}
};
struct Evidence {std::vector<std::vector<float>> local,shared,reduced;std::vector<std::vector<int32_t>> ids;std::vector<std::vector<uint16_t>> output,gate_shard,up_shard,packed,shared_shard,shared_full;size_t experts_checked=0;};
struct Layer {
  int rank,batch;aclrtStream stream=nullptr;HcclComm comm=nullptr;aclmdlRI graph=nullptr;aclScalar *one=nullptr;
  std::unique_ptr<openpangu::Router> router;std::vector<std::unique_ptr<Expert>> experts;
  DeviceTensor input,active,zero,masked,coeff,sum,reduced,shared,final_float,output,packed,gather;
  std::unique_ptr<FloatLinear> sg,su,sd;
  std::vector<std::unique_ptr<Op>> ops;
  std::vector<uint16_t> host_input;std::vector<float> host_active,host_bias,normal_bias;
  Layer(const std::string &dir,int r,int b,const HcclRootInfo &root):rank(r),batch(b),
    input({b,2560},ACL_BF16,2),active({b,1},ACL_FLOAT,4),zero({b,256},ACL_FLOAT,4),masked({b,8},ACL_FLOAT,4),coeff({b,256},ACL_FLOAT,4),
    sum({b,2560},ACL_FLOAT,4),reduced({b,2560},ACL_FLOAT,4),shared({b,2560},ACL_BF16,2),final_float({b,2560},ACL_FLOAT,4),output({b,2560},ACL_BF16,2),packed({b,2048},ACL_BF16,2),gather({8,b,320},ACL_BF16,2){
    check(aclrtCreateStream(&stream),"MoE stream");check(HcclCommInitRootInfo(8,&root,rank,&comm),"MoE HCCL");float v=1;one=aclCreateScalar(&v,ACL_FLOAT);if(!one)throw std::runtime_error("scalar");
    router=std::make_unique<openpangu::Router>(b,2560,256,8,stream);FloatFixture w(dir+"/gate.bin"),bias(dir+"/e_score_correction_bias.bin");router->load(w,bias);normal_bias=bias.values;
    zero.upload(std::vector<float>(b*256,0));
    sg=std::make_unique<FloatLinear>(FloatFixture(dir+"/shared_experts.gate_proj.bin"),b,r,stream);su=std::make_unique<FloatLinear>(FloatFixture(dir+"/shared_experts.up_proj.bin"),b,r,stream);sd=std::make_unique<FloatLinear>(FloatFixture(dir+"/shared_experts.down_proj.bin"),b,r,stream);
    auto op=[this](openpangu::Launcher f,std::function<void(Operation&)>q){ops.push_back(std::make_unique<Op>(f,q));};
    op(aclnnCast,[this](Operation&o){check(aclnnCastGetWorkspaceSize(input.tensor,ACL_FLOAT,router->input.tensor,&o.bytes,&o.executor),"router input cast");});
    op(aclnnMul,[this](Operation&o){check(aclnnMulGetWorkspaceSize(router->scores.tensor,active.tensor,masked.tensor,&o.bytes,&o.executor),"mask routes");});
    op(aclnnScatter,[this](Operation&o){check(aclnnScatterGetWorkspaceSize(zero.tensor,1,router->ids.tensor,masked.tensor,0,coeff.tensor,&o.bytes,&o.executor),"dense route scatter");});
    op(aclnnSwiGlu,[this](Operation&o){check(aclnnSwiGluGetWorkspaceSize(packed.tensor,-1,sd->input.tensor,&o.bytes,&o.executor),"shared SwiGLU");});
    op(aclnnCast,[this](Operation&o){check(aclnnCastGetWorkspaceSize(shared.tensor,ACL_FLOAT,final_float.tensor,&o.bytes,&o.executor),"shared cast");});
    op(aclnnInplaceMul,[this](Operation&o){check(aclnnInplaceMulGetWorkspaceSize(final_float.tensor,active.tensor,&o.bytes,&o.executor),"mask shared");});
    op(aclnnInplaceAdd,[this](Operation&o){check(aclnnInplaceAddGetWorkspaceSize(final_float.tensor,reduced.tensor,one,&o.bytes,&o.executor),"shared plus routed");});
    op(aclnnCast,[this](Operation&o){check(aclnnCastGetWorkspaceSize(final_float.tensor,ACL_BF16,output.tensor,&o.bytes,&o.executor),"MoE output cast");});
    for(int i=0;i<32;++i)experts.push_back(std::make_unique<Expert>(dir,r*32+i,b,stream,coeff,sum,one));
  }
  void gather_projection(FloatLinear &linear,void *destination,int width,int offset){
    linear.run();check(HcclAllGather(linear.output.data,gather.data,batch*linear.n,HCCL_DATA_TYPE_BFP16,comm,stream),"shared TP gather");
    for(int peer=0;peer<8;++peer)for(int b=0;b<batch;++b)copy(static_cast<uint16_t*>(destination)+b*width+offset+peer*linear.n,static_cast<uint16_t*>(gather.data)+(peer*batch+b)*linear.n,linear.n*2,stream);
  }
  void launch(){ops[0]->run(stream);router->launch();ops[1]->run(stream);ops[2]->run(stream);check(aclrtMemsetAsync(sum.data,sum.bytes,0,sum.bytes,stream),"clear local sum");
    for(auto &e:experts)e->run(input.data);
    check(HcclAllReduce(sum.data,reduced.data,batch*2560,HCCL_DATA_TYPE_FP32,HCCL_REDUCE_SUM,comm,stream),"EP combine");
    copy(sg->input.data,input.data,input.bytes,stream);copy(su->input.data,input.data,input.bytes,stream);
    gather_projection(*sg,packed.data,2048,0);gather_projection(*su,packed.data,2048,1024);ops[3]->run(stream);gather_projection(*sd,shared.data,2560,0);
    for(int i=4;i<8;++i)ops[i]->run(stream);
  }
  void set_input(int variant){host_input.resize(batch*2560);host_active.assign(batch,1);host_bias=normal_bias;
    for(size_t j=0;j<host_input.size();++j)host_input[j]=bf16(float(int((j*13+variant*29)%97)-48)/64.f);
    if(variant==2)for(int j=0;j<256;++j)host_bias[j]=(j>=96&&j<104)?100.f:-100.f;
    if(variant==3)for(int b=0;b<batch;++b)host_active[b]=(b%2)?1.f:0.f;
    input.upload(host_input);active.upload(host_active);router->bias.upload(host_bias);
  }
  void validate(int variant,Evidence &e){auto ids=router->ids.read<int32_t>();auto scores=router->scores.read<float>();auto logits=router->logits.read<float>();auto dense=coeff.read<float>();std::vector<float> expected(batch*2560,0);
    for(int b=0;b<batch;++b){double denom=0,minimum=1e30;std::array<bool,256> selected{};for(int j=0;j<8;++j){int id=ids[b*8+j];if(id<0||id>=256||selected[id])throw std::runtime_error("MoE route ids");selected[id]=true;double sig=1/(1+std::exp(-double(logits[b*256+id])));denom+=sig;minimum=std::min(minimum,sig+host_bias[id]);}
      for(int j=0;j<256;++j)if(!selected[j] && 1/(1+std::exp(-double(logits[b*256+j])))+host_bias[j]>minimum+2e-6)throw std::runtime_error("MoE wrong route");
      for(int j=0;j<8;++j){int id=ids[b*8+j];double expected_score=2.5/(1+std::exp(-double(logits[b*256+id])))/(denom+1e-20);if(std::abs(scores[b*8+j]-expected_score)>2e-5)throw std::runtime_error("MoE route weight");if(dense[b*256+id]!=scores[b*8+j]*host_active[b])throw std::runtime_error("masked scatter");}
      for(int j=0;j<256;++j)if(!selected[j] && dense[b*256+j]!=0)throw std::runtime_error("stale scatter");
    }
    for(int i=0;i<32;++i){bool used=false;for(int b=0;b<batch;++b)used|=dense[b*256+rank*32+i]!=0;
      auto &expert=*experts[i];if(used){expert.verify(host_input);++e.experts_checked;}
      auto y=expert.down.read<uint16_t>(5);for(int b=0;b<batch;++b)for(int j=0;j<2560;++j)expected[b*2560+j]+=fp32(y[b*2560+j])*dense[b*256+rank*32+i];
    }
    auto local=sum.read<float>();for(size_t j=0;j<local.size();++j)if(!std::isfinite(local[j])||std::abs(local[j]-expected[j])>1e-4+2e-5*std::abs(expected[j]))throw std::runtime_error("local expert combination");
    sg->verify();su->verify();sd->verify();e.gate_shard.push_back(sg->output.read<uint16_t>());e.up_shard.push_back(su->output.read<uint16_t>());e.packed.push_back(packed.read<uint16_t>());e.shared_shard.push_back(sd->output.read<uint16_t>());e.shared_full.push_back(shared.read<uint16_t>());auto p=packed.read<uint16_t>(),a=sd->input.read<uint16_t>();for(int b=0;b<batch;++b)for(int j=0;j<1024;++j){float g=fp32(p[b*2048+j]);float ref=fp32(bf16(g/(1+std::exp(-g))*fp32(p[b*2048+1024+j])));if(std::abs(fp32(a[b*1024+j])-ref)>1e-4+.016*std::abs(ref))throw std::runtime_error("shared SwiGLU reference");}
    if(variant==2 && rank!=3 && std::any_of(local.begin(),local.end(),[](float x){return x!=0;}))throw std::runtime_error("zero-route rank contributes");
    e.local.push_back(local);auto sh=shared.read<uint16_t>();std::vector<float> sf(sh.size());for(size_t j=0;j<sh.size();++j)sf[j]=fp32(sh[j])*host_active[j/2560];e.shared.push_back(sf);e.reduced.push_back(reduced.read<float>());e.ids.push_back(ids);e.output.push_back(output.read<uint16_t>());
  }
  void run(Evidence &e){for(int v=0;v<4;++v){set_input(v);launch();check(aclrtSynchronizeStream(stream),"MoE eager fence");validate(v,e);}set_input(0);
    check(aclmdlRICaptureBegin(stream,ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),"MoE capture");launch();check(aclmdlRICaptureEnd(stream,&graph),"MoE capture end");
    for(int pass=0;pass<2;++pass)for(int v=0;v<4;++v){set_input(v);check(aclmdlRIExecuteAsync(graph,stream),"MoE replay");check(aclrtSynchronizeStream(stream),"MoE replay fence");if(output.read<uint16_t>()!=e.output[v]||router->ids.read<int32_t>()!=e.ids[v])throw std::runtime_error("MoE graph/eager mismatch");}
  }
  ~Layer(){if(stream)aclrtSynchronizeStream(stream);if(graph)aclmdlRIDestroy(graph);experts.clear();ops.clear();sg.reset();su.reset();sd.reset();router.reset();if(one)aclDestroyScalar(one);if(comm)HcclCommDestroy(comm);if(stream)aclrtDestroyStream(stream);}
};
int main(int argc,char **argv){if(argc!=3){std::cerr<<"usage: openpangu_moe_probe FIXTURES BATCH\n";return 2;}try{int b=std::stoi(argv[2]);if(b!=1&&b!=4)throw std::runtime_error("batch 1 or 4 required");const char* mode=std::getenv("HCCL_OP_EXPANSION_MODE");if(!mode||std::string(mode)!="HOST")throw std::runtime_error("HCCL HOST required");check(aclInit(nullptr),"init");check(aclrtSetDevice(0),"bootstrap");HcclRootInfo root;check(HcclGetRootInfo(&root),"root");std::array<Evidence,8> evidence;std::vector<std::thread> threads;
  for(int r=0;r<8;++r)threads.emplace_back([&,r]{try{check(aclrtSetDevice(r),"rank device");{Layer layer(argv[1],r,b,root);layer.run(evidence[r]);}check(aclrtResetDevice(r),"rank reset");}catch(const std::exception&e){std::cerr<<"rank "<<r<<": "<<e.what()<<'\n';std::_Exit(1);}});
  for(auto&t:threads)t.join();
  size_t checked_values=0,expert_checks=0;
  for(auto&e:evidence)expert_checks+=e.experts_checked;
  for(int v=0;v<4;++v)for(int r=0;r<8;++r){
    for(int token=0;token<b;++token)for(int peer=0;peer<8;++peer){
      for(int j=0;j<128;++j){if(evidence[r].packed[v][token*2048+peer*128+j]!=evidence[peer].gate_shard[v][token*128+j] || evidence[r].packed[v][token*2048+1024+peer*128+j]!=evidence[peer].up_shard[v][token*128+j])throw std::runtime_error("shared gate/up TP gather layout");}
      for(int j=0;j<320;++j)if(evidence[r].shared_full[v][token*2560+peer*320+j]!=evidence[peer].shared_shard[v][token*320+j])throw std::runtime_error("shared down TP gather layout");
    }
    if(evidence[r].ids[v]!=evidence[0].ids[v])throw std::runtime_error("rank router divergence");
    for(size_t j=0;j<evidence[r].reduced[v].size();++j){double sum=0;for(auto&e:evidence)sum+=e.local[v][j];double actual=evidence[r].reduced[v][j];if(!std::isfinite(actual)||std::abs(actual-sum)>1e-4+2e-5*std::abs(sum))throw std::runtime_error("EP reference sum");float ref=fp32(bf16(float(actual)+evidence[r].shared[v][j]));if(fp32(evidence[r].output[v][j])!=ref)throw std::runtime_error("shared plus routed result");++checked_values;}}
  check(aclFinalize(),"finalize");std::cout<<"{\"status\":\"passed\",\"batch\":"<<b<<",\"tp\":8,\"ep\":8,\"loaded_experts\":256,\"cases\":4,\"graph_passes\":2,\"active_expert_checks\":"<<expert_checks<<",\"combined_values_checked\":"<<checked_values<<",\"execution\":\"dense masked correctness baseline\"}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
