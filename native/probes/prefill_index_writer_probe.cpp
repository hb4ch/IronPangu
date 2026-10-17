#include "../src/resident_index.h"
#include <iostream>
int main(){using namespace inferfabric;try{
  prepared::check(aclInit(nullptr),"init");prepared::check(aclrtSetDevice(0),"device");
  size_t checked=0,cases=0;
  for(bool quant:{false,true})for(int rows:{1,17,4096}) {
    prepared::Context context;
    resident::IndexState state(8192,2,false,quant,rows);
    openpangu::DeviceTensor source({rows,128},quant?ACL_INT8:ACL_BF16,quant?1:2),
        scale({rows},ACL_FLOAT16,2),positions({rows*4},ACL_INT32,4);
    std::vector<uint8_t> previous(state.keys.bytes,0),previous_scales(quant?state.scales->bytes:0,0);
    auto read=[&](const openpangu::DeviceTensor &tensor){
      void *host=context.output_host(tensor.bytes);
      {resident::TransferScope scope(resident::TransferRole::RequestOutput);
        prepared::check(aclrtMemcpyAsync(host,tensor.bytes,tensor.data,tensor.bytes,ACL_MEMCPY_DEVICE_TO_HOST,context.stream),"index fixture output");}
      context.complete();auto *bytes=static_cast<uint8_t *>(host);return std::vector<uint8_t>(bytes,bytes+tensor.bytes);
    };
    for(int start:{63,4096}) {
      std::vector<uint8_t> values(source.bytes);std::vector<uint16_t> scales(rows);
      std::vector<int32_t> slots(rows*4,-999);std::vector<bool> touched(16384,false);
      for(int row=0;row<rows;++row) {
        int logical=start+row;
        slots[row*4+2]=8192+((logical/64)*37%128)*64+logical%64;touched[slots[row*4+2]]=true;
        scales[row]=uint16_t(0x3c00+(row%16)*16);
        for(int column=0;column<128;++column) {
          if(quant)values[row*128+column]=uint8_t((logical*13+column)%255);
          else {uint16_t value=uint16_t(0x3f00+(logical+column)%127);std::memcpy(values.data()+(row*128+column)*2,&value,2);}
        }
      }
      resident::control_upload(source,values);resident::control_upload(scale,scales);resident::control_upload(positions,slots);context.complete();
      auto *indices=context.strided_descriptor(static_cast<int32_t *>(positions.data)+2,rows*4-2,{rows},{4},ACL_INT32);
      if(quant)state.append_quantized(resident::Tensor::alias(source,ACL_INT8),resident::Tensor::alias(scale,ACL_FLOAT16),indices,rows);
      else state.append(resident::Tensor::alias(source,ACL_BF16),indices,rows);
      auto actual=read(state.keys);auto actual_scales=quant?read(*state.scales):std::vector<uint8_t>{};
      size_t width=128*(quant?1:2);
      for(int physical=0;physical<16384;++physical)if(!touched[physical]) {
        if(std::memcmp(actual.data()+physical*width,previous.data()+physical*width,width))throw std::runtime_error("unwritten index row changed");
        if(quant && std::memcmp(actual_scales.data()+physical*2,previous_scales.data()+physical*2,2))throw std::runtime_error("unwritten index scale changed");
      }
      for(int row=0;row<rows;++row) {
        auto physical=slots[row*4+2];
        if(std::memcmp(actual.data()+physical*width,values.data()+row*width,width))throw std::runtime_error("index key mismatch");
        if(quant && std::memcmp(actual_scales.data()+physical*2,&scales[row],2))throw std::runtime_error("index scale mismatch");
        checked+=128;
      }
      previous=std::move(actual);previous_scales=std::move(actual_scales);
    }
    std::cout<<"{\"quant\":"<<(quant?"true":"false")<<",\"rows\":"<<rows<<",\"audit\":";
    context.write(std::cout);std::cout<<"}\n";++cases;
  }
  prepared::check(aclrtResetDevice(0),"reset");prepared::check(aclFinalize(),"finalize");
  std::cout<<"{\"status\":\"passed\",\"cases\":"<<cases<<",\"checked_keys\":"<<checked<<"}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
