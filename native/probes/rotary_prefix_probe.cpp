#include "../src/resident_rotary.h"
#include <cmath>
#include <iostream>

static uint16_t bf(float f){uint32_t b;std::memcpy(&b,&f,4);return uint16_t((b+0x7fff+((b>>16)&1))>>16);}
static float fp(uint16_t v){uint32_t b=uint32_t(v)<<16;float f;std::memcpy(&f,&b,4);return f;}
int main(){using namespace inferfabric;try{
  prepared::check(aclInit(nullptr),"init");prepared::check(aclrtSetDevice(0),"device");
  size_t checked=0,replays=0;float max_error=0;
  for(int heads:{1,6,24})for(int tokens:{1,2,4,8,16}) {
    prepared::Context context;
    openpangu::DeviceTensor input({tokens,heads,128},ACL_BF16,2),
      cosine({tokens,1,1,64},ACL_BF16,2),sine({tokens,1,1,64},ACL_BF16,2);
    std::vector<uint16_t> x(tokens*heads*128),c(tokens*64),s(tokens*64);
    auto load=[&](int position){
      for(size_t i=0;i<x.size();++i)x[i]=bf(float(int((i*37+position)%257)-128)/64);
      for(int row=0;row<tokens;++row)for(int d=0;d<64;++d){
        double phase=(position+row)*std::pow(6400000.,-2.*(d%32)/64);
        c[row*64+d]=bf(float(std::cos(phase)));s[row*64+d]=bf(float(std::sin(phase)));
      }
      // Qualification fixtures only: production controls contain no activation.
      resident::control_upload(input,x);resident::control_upload(cosine,c);resident::control_upload(sine,s);
      context.complete();
    };
    std::vector<resident::Tensor> outputs;
    auto launch=[&]{
      auto source=resident::Tensor::alias(input,ACL_BF16);
      auto prefix=resident::column_slice(source,tokens*heads,0,64);
      auto tail=resident::column_slice(source,tokens*heads,64,64);
      std::vector<resident::Tensor> result;
      result.push_back(resident::rope_bf16(prefix,cosine.tensor,sine.tensor,tokens));
      result.push_back(tail);return result;
    };
    auto read=[&](const resident::Tensor &v){
      auto *host=context.output_host(v.bytes());
      {resident::TransferScope scope(resident::TransferRole::RequestOutput);
        prepared::check(aclrtMemcpyAsync(host,v.bytes(),v.data(),v.bytes(),ACL_MEMCPY_DEVICE_TO_HOST,context.stream),"test output");}
      context.complete();return std::vector<uint16_t>(static_cast<uint16_t *>(host),static_cast<uint16_t *>(host)+v.size());
    };
    auto verify=[&]{
      auto tail=read(outputs.back());
      auto actual=read(outputs.front());
      for(int row=0;row<tokens;++row){
        for(int h=0;h<heads;++h)for(int d=0;d<64;++d){
          size_t offset=(row*heads+h)*128;
          float a=fp(x[offset+d]),b=fp(x[offset+(d+32)%64]);
          float expected=fp(bf(a*fp(c[row*64+d])+(d<32?-b:b)*fp(s[row*64+d])));
          float error=std::abs(fp(actual[(row*heads+h)*64+d])-expected);max_error=std::max(max_error,error);
          if(actual[(row*heads+h)*64+d]!=bf(expected))throw std::runtime_error("BF16 rotary oracle mismatch");
          if(tail[(row*heads+h)*64+d]!=x[offset+64+d])throw std::runtime_error("rotary tail changed");
          ++checked;
        }
      }
    };
    load(0);outputs=launch();verify();outputs.clear();
    aclmdlRI graph=nullptr;
    prepared::check(aclmdlRICaptureBegin(context.stream,ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),"capture");
    outputs=launch();prepared::check(aclmdlRICaptureEnd(context.stream,&graph),"capture end");
    for(int position:{1,511,2048,65536-tokens,0}){
      load(position);prepared::check(aclmdlRIExecuteAsync(graph,context.stream),"replay");verify();++replays;
    }
    prepared::check(aclmdlRIDestroy(graph),"graph destroy");
  }
  prepared::check(aclrtResetDevice(0),"reset");prepared::check(aclFinalize(),"finalize");
  std::cout<<"{\"status\":\"passed\",\"checked_rotary_and_tail_pairs\":"<<checked
    <<",\"graph_replays\":"<<replays<<",\"max_abs_error\":"<<max_error<<"}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
