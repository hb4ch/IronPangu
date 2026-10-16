#include "openpangu_probe_common.h"
#include "../src/openpangu_ops.h"
#include "aclnn_ai_infra_kv_quant_sparse_flash_attention.h"
#include "cache_quantization.h"
#include <aclnn_ai_infra_kv_rms_norm_rope_cache.h>
#include <numeric>
using openpangu::DeviceTensor;
using openpangu::Operation;
int main() {
 try {
  check(aclInit(nullptr),"init");check(aclrtSetDevice(0),"device");
  {
   aclrtStream stream=nullptr;check(aclrtCreateStream(&stream),"stream");
   DeviceTensor x({512,1,1,576},ACL_BF16,2),gamma({512},ACL_BF16,2),cos({512,1,1,64},ACL_BF16,2),sin({512,1,1,64},ACL_BF16,2),index({512},ACL_INT64,8);
   DeviceTensor cache({16,32,1,656},ACL_INT8,1),rope({512,1,1,64},ACL_BF16,2),latent({512,1,1,512},ACL_BF16,2);
   std::vector<uint16_t> hostx(512*576),hostgamma(512,bf16(1)),hostcos(512*64,bf16(.8f)),hostsin(512*64,bf16(.6f));
   std::vector<int64_t> slots(512);std::iota(slots.begin(),slots.end(),0);
   for(int r=0;r<512;++r)for(int d=0;d<576;++d)hostx[r*576+d]=bf16(r==0?0:float((r*7+d*11)%97-48)*(1+d/128)/128);
   x.upload(hostx);gamma.upload(hostgamma);cos.upload(hostcos);sin.upload(hostsin);index.upload(slots);
   char mode[]="PA",rotary[]="half",quant[]="pertile128";
   Operation writer;
   check(aclnnAiInfraKvRmsNormRopeCacheGetWorkspaceSize(x.tensor,gamma.tensor,cos.tensor,sin.tensor,index.tensor,nullptr,cache.tensor,nullptr,nullptr,nullptr,nullptr,nullptr,1e-5,mode,rotary,quant,true,rope.tensor,latent.tensor,&writer.bytes,&writer.executor),"C8 writer prepare");
   writer.allocate();writer.run(aclnnAiInfraKvRmsNormRopeCache,stream);check(aclrtSynchronizeStream(stream),"writer fence");
   auto packed=cache.read<uint8_t>();double scale_error=0;int quant_error=0;
   for(int r=0;r<512;++r) {
    double square=0;for(int d=0;d<512;++d){double v=fp32(hostx[r*576+d]);square+=v*v;}
    double norm=1/std::sqrt(square/512+1e-5);
    for(int g=0;g<4;++g) {
     float scale;std::memcpy(&scale,packed.data()+r*656+640+g*4,4);
     double maximum=0;for(int d=g*128;d<(g+1)*128;++d)maximum=std::max(maximum,std::abs(fp32(hostx[r*576+d])*norm));
     double expected_scale=maximum/127;scale_error=std::max(scale_error,std::abs(scale-expected_scale));
     if(std::abs(scale-expected_scale)>1e-5+expected_scale*.002)throw std::runtime_error("C8 tile scale mismatch");
     for(int d=g*128;d<(g+1)*128;++d) {
      int actual=int(int8_t(packed[r*656+d]));int expected=expected_scale==0?0:int(std::nearbyint(fp32(hostx[r*576+d])*norm/expected_scale));
      quant_error=std::max(quant_error,std::abs(actual-expected));if(std::abs(actual-expected)>1)throw std::runtime_error("C8 latent quantization mismatch");
     }
    }
    for(int d=0;d<64;++d) {
     float a=fp32(hostx[r*576+512+d]),b=fp32(hostx[r*576+512+(d+32)%64]);
     auto expected=bf16(a*fp32(hostcos[r*64+d])+(d<32?-b:b)*fp32(hostsin[r*64+d]));uint16_t actual;
     std::memcpy(&actual,packed.data()+r*656+512+d*2,2);
     if(actual!=expected)throw std::runtime_error("C8 rotary is not expected BF16");
    }
   }
   DeviceTensor q({1,6,576},ACL_BF16,2),indices({1,1,512},ACL_INT32,4),length({1},ACL_INT32,4),position({1},ACL_INT64,8),qlen({1},ACL_INT32,4),table({1,16},ACL_INT32,4),sink({128,1,576},ACL_BF16,2),sinkvalues({128,1,512},ACL_BF16,2),out({1,6,512},ACL_BF16,2);
   std::vector<uint16_t> hostq(6*576),hosts(128*576),sv(128*512);std::vector<int32_t> blocks(16);std::iota(blocks.begin(),blocks.end(),0);
   for(size_t i=0;i<hostq.size();++i)hostq[i]=bf16(float(int(i*13%101)-50)/128);
   for(size_t i=0;i<hosts.size();++i)hosts[i]=bf16(float(int(i*19%103)-51)/128);
   for(int r=0;r<128;++r)for(int d=0;d<512;++d)sv[r*512+d]=hosts[r*576+d];
   q.upload(hostq);sink.upload(hosts);sinkvalues.upload(sv);table.upload(blocks);qlen.upload(std::vector<int32_t>{1});
   auto prepare=[&](Operation &op){char lq[]="TND",lk[]="PA_BSND";check(aclnnAiInfraKvQuantSparseFlashAttentionGetWorkspaceSize(q.tensor,cache.tensor,cache.tensor,indices.tensor,nullptr,nullptr,table.tensor,qlen.tensor,length.tensor,sink.tensor,sinkvalues.tensor,1/std::sqrt(192.),2,2,1,lq,lk,3,INT64_MAX,INT64_MAX,2,1,128,64,out.tensor,&op.bytes,&op.executor),"packed attention prepare");op.allocate();};
   auto controls=[&](int pos){position.upload(std::vector<int64_t>{pos});};
   auto metadata=[&](){if_packed_attention_metadata(position.data,indices.data,length.data,512,stream);};
   auto unpack=[&](int r,int d){if(d<512){float scale;std::memcpy(&scale,packed.data()+r*656+640+d/128*4,4);return int(int8_t(packed[r*656+d]))*scale;}uint16_t v;std::memcpy(&v,packed.data()+r*656+512+(d-512)*2,2);return fp32(v);};
   double max_error=0;size_t checked_values=0;
   auto verify=[&](int pos){auto actual=out.read<uint16_t>();auto ids=indices.read<int32_t>();if(length.read<int32_t>()[0]!=pos+1)throw std::runtime_error("sequence length mismatch");for(int i=0;i<512;++i)if(ids[i]!=(i<=pos?i:-1))throw std::runtime_error("indices mismatch");
    for(int h=0;h<6;++h){std::vector<double> scores(pos+1+128);double mx=-1e30,total=0;
     for(int r=0;r<int(scores.size());++r){double dot=0;for(int d=0;d<576;++d)dot+=fp32(hostq[h*576+d])*(r<=pos?unpack(r,d):fp32(hosts[(r-pos-1)*576+d]));scores[r]=dot/std::sqrt(192.);mx=std::max(mx,scores[r]);}
     for(auto &s:scores){s=std::exp(s-mx);total+=s;}
     for(int d=0;d<512;++d){double expected=0;for(int r=0;r<int(scores.size());++r)expected+=scores[r]/total*(r<=pos?unpack(r,d):fp32(hosts[(r-pos-1)*576+d]));double error=std::abs(fp32(actual[h*512+d])-expected);max_error=std::max(max_error,error);if(!std::isfinite(error)||error>.003+.02*std::abs(expected))throw std::runtime_error("packed attention CPU mismatch");++checked_values;}
    }
   };
   std::vector<std::vector<uint16_t>> eager;
   for(int pos:{0,31,32,511}){controls(pos);metadata();Operation op;prepare(op);op.run(aclnnAiInfraKvQuantSparseFlashAttention,stream);check(aclrtSynchronizeStream(stream),"attention fence");verify(pos);eager.push_back(out.read<uint16_t>());}
   Operation op;prepare(op);aclmdlRI graph=nullptr;check(aclmdlRICaptureBegin(stream,ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),"capture");metadata();op.run(aclnnAiInfraKvQuantSparseFlashAttention,stream);check(aclmdlRICaptureEnd(stream,&graph),"capture end");
   for(int pass=0;pass<2;++pass){int i=0;for(int pos:{0,31,32,511}){controls(pos);check(aclmdlRIExecuteAsync(graph,stream),"replay");check(aclrtSynchronizeStream(stream),"replay fence");verify(pos);if(out.read<uint16_t>()!=eager[i++])throw std::runtime_error("packed attention graph mismatch");}}
   check(aclmdlRIDestroy(graph),"destroy graph");check(aclrtDestroyStream(stream),"destroy stream");
   std::cout<<"{\"status\":\"passed\",\"cache_rows\":512,\"scale_max_abs_error\":"<<scale_error<<",\"quant_max_integer_error\":"<<quant_error<<",\"attention_values_checked\":"<<checked_values<<",\"attention_max_abs_error\":"<<max_error<<"}\n";
  }
  check(aclrtResetDevice(0),"reset");check(aclFinalize(),"finalize");
 }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
