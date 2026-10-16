#include "../src/openpangu_ops.h"
#include "cache_quantization.h"
#include <iostream>
#include <cstring>
using namespace openpangu;
static uint16_t bf(float f){uint32_t u;std::memcpy(&u,&f,4);u+=0x7fff+((u>>16)&1);return u>>16;}
static float fp(uint16_t b){uint32_t u=uint32_t(b)<<16;float f;std::memcpy(&f,&u,4);return f;}
int main(){try{
 checked(aclInit(nullptr),"init");checked(aclrtSetDevice(0),"device");aclrtStream stream;checked(aclrtCreateStream(&stream),"stream");
 for(unsigned bits:{8,4}){
  const unsigned width=576,rows=512;
  DeviceTensor input({width},ACL_BF16,2),packed({rows*width*bits/8},ACL_UINT8,1),scales({rows},ACL_FLOAT,4),position({1},ACL_INT64,8),output({rows*width},ACL_BF16,2);
  std::vector<uint16_t> x(width);for(unsigned j=0;j<width;++j)x[j]=bf(float(int(j%61)-30)/13.f);
  input.upload(x);packed.upload(std::vector<uint8_t>(packed.bytes,0xa5));scales.upload(std::vector<float>(rows,-1));
  for(int64_t row:{0,1,31,511}){
   position.upload(std::vector<int64_t>{row});
   if_cache_quantize(input.data,packed.data,scales.data,position.data,width,rows,bits,stream);
   checked(aclrtSynchronizeStream(stream),"quantize");
   auto p=packed.read<uint8_t>();auto s=scales.read<float>();float max=0;for(auto v:x)max=std::max(max,std::abs(fp(v)));float expected=max/(bits==8?127:7);
   if(std::abs(s[row]-expected)>1e-6)throw std::runtime_error("scale mismatch");
   for(unsigned j=0;j<width;++j){int q=std::round(fp(x[j])/expected);unsigned byte=p[row*width*bits/8+j*bits/8];int actual=bits==8?int(int8_t(byte)):int((byte>>((j%2)*4))&15);if(bits==4&&actual>=8)actual-=16;if(q!=actual)throw std::runtime_error("packed mismatch");}
  }
  // Graph replay must read a changed device position, including the zero row.
  position.upload(std::vector<int64_t>{0});
  aclmdlRI graph=nullptr;checked(aclmdlRICaptureBegin(stream,ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),"capture");
  if_cache_quantize(input.data,packed.data,scales.data,position.data,width,rows,bits,stream);
  if_cache_dequantize(packed.data,scales.data,output.data,position.data,width,rows,bits,stream);
  checked(aclmdlRICaptureEnd(stream,&graph),"capture end");
  for(int64_t row:{0,1}){
   if(row==1){std::fill(x.begin(),x.end(),0);input.upload(x);}
   position.upload(std::vector<int64_t>{row});checked(aclmdlRIExecuteAsync(graph,stream),"replay");checked(aclrtSynchronizeStream(stream),"fence");
   auto y=output.read<uint16_t>();auto p=packed.read<uint8_t>();auto s=scales.read<float>();
   for(unsigned r=0;r<rows;++r)for(unsigned j=0;j<width;++j){int q=0;if(r<=row){unsigned byte=p[r*width*bits/8+j*bits/8];q=bits==8?int(int8_t(byte)):int((byte>>((j%2)*4))&15);if(bits==4&&q>=8)q-=16;}auto expected=bf(r<=row?q*s[r]:0);if(y[r*width+j]!=expected)throw std::runtime_error("dequant replay mismatch");}
  }
  checked(aclmdlRIDestroy(graph),"destroy");std::cout<<"{\"bits\":"<<bits<<",\"packed_bytes\":"<<packed.bytes<<",\"scale_bytes\":"<<scales.bytes<<",\"graph_replay\":true,\"pass\":true}\n";
 }
 checked(aclrtDestroyStream(stream),"stream destroy");return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
