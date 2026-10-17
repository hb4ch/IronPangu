#include "../src/resident_head.h"
#include <iostream>
#include <cmath>

static uint16_t bf(float f){uint32_t b;std::memcpy(&b,&f,4);return uint16_t((b+0x7fff+((b>>16)&1))>>16);}
static float fp(uint16_t v){uint32_t b=uint32_t(v)<<16;float f;std::memcpy(&f,&b,4);return f;}
int main(){using namespace inferfabric;try{
  prepared::check(aclInit(nullptr),"init");prepared::check(aclrtSetDevice(0),"device");
  size_t checked=0,cases=0;
  for(int mode:{0,1,2})for(int tokens:{1,2,17,127,4096})for(bool dense:{false,true}) {
    if(dense && tokens!=17)continue;
    const bool query=mode!=2,replicated=mode==1;
    const int heads=6,k=query?128:512,n=query?512:128;
    const int head_stride=query?192:512,token_stride=replicated?9216:heads*head_stride;
    const int offset=replicated?7*1152:0,weight_offset=query?0:65536;
    prepared::Context context;
    openpangu::DeviceTensor input({tokens,token_stride},ACL_BF16,2),weight({1536,512},ACL_BF16,2);
    std::vector<uint16_t> x(tokens*token_stride),w(1536*512,0);
    for(size_t i=0;i<x.size();++i)x[i]=bf(float(int(i*13%63)-31)/16);
    for(int h=0;h<heads;++h)for(int column=0;column<n;++column)for(int inner=0;inner<k;++inner) {
      float value=dense?float((h+column+inner)%3-1)/16:(inner==column%k?(h%2?-1.f:1.f):0.f);
      w[h*131072+weight_offset+(query?inner*512+column:column*512+inner)]=bf(value);
    }
    resident::control_upload(input,x);resident::control_upload(weight,w);context.complete();
    auto *a=context.strided_descriptor(static_cast<uint16_t *>(input.data)+offset,x.size()-offset,
        {heads,tokens,k},{head_stride,token_stride,1},ACL_BF16);
    auto *b=context.strided_descriptor(static_cast<uint16_t *>(weight.data)+weight_offset,w.size()-weight_offset,
        {heads,k,n},{131072,query?512:1,query?1:512},ACL_BF16);
    auto projected=resident::head_matmul(a,b,heads,tokens,n);
    auto output=resident::head_to_token(projected,heads,tokens,n);
    if(tokens==1 && output.data()!=projected.data())throw std::runtime_error("decode layout is not an alias");
    void *host=context.output_host(output.bytes());
    {resident::TransferScope scope(resident::TransferRole::RequestOutput);
      prepared::check(aclrtMemcpyAsync(host,output.bytes(),output.data(),output.bytes(),ACL_MEMCPY_DEVICE_TO_HOST,context.stream),"test output");}
    context.complete();auto *actual=static_cast<uint16_t *>(host);
    for(int row=0;row<tokens;++row)for(int h=0;h<heads;++h)for(int column=0;column<n;++column) {
      float expected=0;
      if(dense)for(int inner=0;inner<k;++inner)
        expected+=fp(x[row*token_stride+offset+h*head_stride+inner])*fp(w[h*131072+weight_offset+(query?inner*512+column:column*512+inner)]);
      else expected=fp(x[row*token_stride+offset+h*head_stride+column%k])*(h%2?-1.f:1.f);
      if(fp(actual[(row*heads+h)*n+column])!=fp(bf(expected)))
        throw std::runtime_error("head projection oracle mismatch mode="+std::to_string(mode)+" tokens="+std::to_string(tokens));
      ++checked;
    }
    std::cout<<"{\"mode\":"<<mode<<",\"tokens\":"<<tokens<<",\"dense\":"<<(dense?"true":"false")<<",\"audit\":";
    context.write(std::cout);std::cout<<"}\n";++cases;
  }
  prepared::check(aclrtResetDevice(0),"reset");prepared::check(aclFinalize(),"finalize");
  std::cout<<"{\"status\":\"passed\",\"cases\":"<<cases<<",\"checked_values\":"<<checked<<"}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
