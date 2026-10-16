#include "kernel_operator.h"
#include <cstdint>
// Portable storage kernels. INT4 uses signed two's-complement nibbles, low first.
// Each row has one FP32 scale. Position is device control state for graph replay.
__aicore__ inline float read_bf16(uint16_t v) {
  union { uint32_t u; float f; } x; x.u=uint32_t(v)<<16; return x.f;
}
__aicore__ inline uint16_t write_bf16(float v) {
  union { uint32_t u; float f; } x; x.f=v;
  x.u += 0x7fff + ((x.u>>16)&1); return uint16_t(x.u>>16);
}
extern "C" __global__ __aicore__ void cache_quantize_kernel(
    GM_ADDR input_addr, GM_ADDR packed_addr, GM_ADDR scales_addr,
    GM_ADDR position_addr, uint32_t width, uint32_t capacity, uint32_t bits) {
  auto packed=(__gm__ uint8_t*)packed_addr; auto scales=(__gm__ float*)scales_addr; auto position=(__gm__ int64_t*)position_addr;
  auto input=(__gm__ uint16_t*)input_addr;
  int64_t row=*position;
  if(row<0 || row>=capacity) return;
  float maximum=0;
  for(uint32_t j=0;j<width;++j) {
    float x=read_bf16(input[j]); float a=x<0?-x:x; if(a>maximum)maximum=a;
  }
  int limit=bits==8?127:7;
  float scale=maximum==0?1.0f:maximum/float(limit);
  scales[row]=scale;
  uint32_t stride=width*bits/8;
  for(uint32_t j=0;j<width;j+=8/bits) {
    uint8_t byte=0;
    for(uint32_t k=0;k<8/bits;++k) {
      float value=read_bf16(input[j+k])/scale;
      // Symmetric nearest rounding, ties away from zero, explicitly specified.
      int q=int(value+(value>=0?0.5f:-0.5f));
      q=q>limit?limit:(q < -limit?-limit:q);
      byte |= uint8_t(q & (bits==8?255:15)) << (k*bits);
    }
    packed[row*stride+j*bits/8]=byte;
  }
}
extern "C" __global__ __aicore__ void cache_dequantize_kernel(
    GM_ADDR packed_addr, GM_ADDR scales_addr, GM_ADDR output_addr,
    GM_ADDR position_addr, uint32_t width, uint32_t capacity, uint32_t bits) {
  auto packed=(__gm__ uint8_t*)packed_addr; auto scales=(__gm__ float*)scales_addr; auto position=(__gm__ int64_t*)position_addr;
  auto output=(__gm__ uint16_t*)output_addr;
  int64_t last=*position;
  for(uint32_t row=AscendC::GetBlockIdx();row<capacity;row+=AscendC::GetBlockNum()) {
    float scale=row<=last?scales[row]:0;
    for(uint32_t j=0;j<width;++j) {
      int q=0;
      if(row<=last) {
        uint8_t byte=packed[row*(width*bits/8)+j*bits/8];
        q=bits==8?int(int8_t(byte)):int((byte>>((j%2)*4))&15);
        if(bits==4 && q>=8)q-=16;
      }
      output[row*width+j]=write_bf16(float(q)*scale);
    }
  }
}
extern "C" void if_cache_quantize(void *input,void *packed,void *scales,void *position,
    uint32_t width,uint32_t capacity,uint32_t bits,void *stream) {
  cache_quantize_kernel<<<1,nullptr,stream>>>((uint8_t*)input,(uint8_t*)packed,
      (uint8_t*)scales,(uint8_t*)position,width,capacity,bits);
}
extern "C" void if_cache_dequantize(void *packed,void *scales,void *output,void *position,
    uint32_t width,uint32_t capacity,uint32_t bits,void *stream) {
  cache_dequantize_kernel<<<32,nullptr,stream>>>((uint8_t*)packed,(uint8_t*)scales,
      (uint8_t*)output,(uint8_t*)position,width,capacity,bits);
}

// Causal candidates are generated from device control state, once per token.
extern "C" __global__ __aicore__ void packed_attention_metadata_kernel(
    GM_ADDR position_addr, GM_ADDR indices_addr, GM_ADDR length_addr, uint32_t capacity) {
  int64_t position=*((__gm__ int64_t*)position_addr);
  auto indices=(__gm__ int32_t*)indices_addr;
  *((__gm__ int32_t*)length_addr)=int32_t(position+1);
  for(uint32_t i=0;i<capacity;++i)indices[i]=int64_t(i)<=position?int32_t(i):-1;
}
extern "C" void if_packed_attention_metadata(void *position,void *indices,void *length,uint32_t capacity,void *stream) {
  packed_attention_metadata_kernel<<<1,nullptr,stream>>>((uint8_t*)position,(uint8_t*)indices,(uint8_t*)length,capacity);
}
