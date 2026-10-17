#include "../src/resident_cache_writer.h"
#include <iostream>
#include <cmath>

static uint16_t bf(float f){uint32_t b;std::memcpy(&b,&f,4);return uint16_t((b+0x7fff+((b>>16)&1))>>16);}
static float fp(uint16_t v){uint32_t b=uint32_t(v)<<16;float f;std::memcpy(&f,&b,4);return f;}
int main(){using namespace inferfabric;try{
  prepared::check(aclInit(nullptr),"init");prepared::check(aclrtSetDevice(0),"device");
  size_t launches=0,checked=0;double max_scale_error=0,max_bf16_error=0;
  for(bool packed:{false,true})for(int rows:{1,2,17,127,4096}) {
    prepared::Context context;
    const int capacity=8192,page=packed?32:16,stride=packed?656:1152;
    openpangu::DeviceTensor x({rows,1,1,576},ACL_BF16,2),gamma({512},ACL_BF16,2),
      cosine({rows,1,1,64},ACL_BF16,2),sine({rows,1,1,64},ACL_BF16,2),positions({rows*2},ACL_INT64,8),
      cache({capacity/page,page,1,packed?656:576},packed?ACL_INT8:ACL_BF16,packed?1:2);
    std::vector<uint16_t> input(rows*576),g(512,bf(1)),c(rows*64,bf(.8f)),s(rows*64,bf(.6f));
    std::vector<uint8_t> before(capacity*stride,0xa5);
    resident::control_upload(gamma,g);resident::control_upload(cosine,c);resident::control_upload(sine,s);
    resident::control_upload(cache,before);context.complete();
    for(int start:{0,4096}) {
      std::vector<int64_t> slots(rows*2,-777);std::vector<bool> touched(capacity,false);
      for(int row=0;row<rows;++row) {
        int logical=start+row;
        slots[row*2]=((logical/page)*37%(capacity/page))*page+logical%page;
        touched[slots[row*2]]=true;
        for(int d=0;d<576;++d)input[row*576+d]=bf(float((logical*7+d*11)%97-48)*(1+d/128)/128);
      }
      resident::control_upload(x,input);resident::control_upload(positions,slots);context.complete();
      auto *position=context.strided_descriptor(positions.data,rows*2,{rows},{2},ACL_INT64);
      resident::normalized_cache_append(resident::Tensor::alias(x,ACL_BF16),gamma.tensor,cosine.tensor,
          sine.tensor,position,cache.tensor,rows,packed);++launches;
      void *host=context.output_host(cache.bytes);
      {resident::TransferScope scope(resident::TransferRole::RequestOutput);
        prepared::check(aclrtMemcpyAsync(host,cache.bytes,cache.data,cache.bytes,ACL_MEMCPY_DEVICE_TO_HOST,context.stream),"cache fixture output");}
      context.complete();auto *actual=static_cast<uint8_t *>(host);
      for(int physical=0;physical<capacity;++physical)if(!touched[physical])
        if(std::memcmp(actual+physical*stride,before.data()+physical*stride,stride))throw std::runtime_error("unwritten cache row changed");
      for(int row=0;row<rows;++row) {
        const auto *out=actual+slots[row*2]*stride;
        double square=0;for(int d=0;d<512;++d){double value=fp(input[row*576+d]);square+=value*value;}
        double norm=1/std::sqrt(square/512+1e-5);
        for(int tile=0;tile<4;++tile) {
          double maximum=0;for(int d=tile*128;d<(tile+1)*128;++d)maximum=std::max(maximum,std::abs(fp(input[row*576+d])*norm));
          double scale=maximum/127;
          if(packed){float value;std::memcpy(&value,out+640+tile*4,4);max_scale_error=std::max(max_scale_error,std::abs(value-scale));
            if(std::abs(value-scale)>1e-5+.002*scale)throw std::runtime_error("C8 scale mismatch");}
          for(int d=tile*128;d<(tile+1)*128;++d) {
            double expected=fp(input[row*576+d])*norm;
            if(packed){int quant=scale==0?0:int(std::nearbyint(expected/scale));if(std::abs(int(int8_t(out[d]))-quant)>1)throw std::runtime_error("C8 latent mismatch");}
            else {uint16_t value;std::memcpy(&value,out+d*2,2);double error=std::abs(fp(value)-expected);max_bf16_error=std::max(max_bf16_error,error);
              if(error>.0078125*std::max(1.,std::abs(expected)))throw std::runtime_error("BF16 latent mismatch");}
            ++checked;
          }
        }
        for(int d=0;d<64;++d) {
          float a=fp(input[row*576+512+d]),b=fp(input[row*576+512+(d+32)%64]);
          auto expected=bf(a*fp(c[row*64+d])+(d<32?-b:b)*fp(s[row*64+d]));uint16_t value;
          std::memcpy(&value,out+(packed?512:1024)+d*2,2);
          if(value!=expected)throw std::runtime_error("BF16 rotary mismatch");
          ++checked;
        }
      }
      before.assign(actual,actual+cache.bytes);
    }
    std::cout<<"{\"packed\":"<<(packed?"true":"false")<<",\"rows\":"<<rows<<",\"audit\":";
    context.write(std::cout);std::cout<<"}\n";
  }
  prepared::check(aclrtResetDevice(0),"reset");prepared::check(aclFinalize(),"finalize");
  std::cout<<"{\"status\":\"passed\",\"batch_launches\":"<<launches<<",\"checked_values\":"<<checked
    <<",\"max_scale_error\":"<<max_scale_error<<",\"max_bf16_error\":"<<max_bf16_error<<"}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
