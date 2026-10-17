#include "../src/resident_convolution.h"
#include "../include/sequence_batch.h"
#include <iostream>
#include <cmath>
static uint16_t bf(float v){uint32_t n;std::memcpy(&n,&v,4);return uint16_t((n+0x7fff+((n>>16)&1))>>16);}
static float fp(uint16_t v){uint32_t n=uint32_t(v)<<16;float f;std::memcpy(&f,&n,4);return f;}
int main(){using namespace inferfabric;try{
  prepared::check(aclInit(nullptr),"init");prepared::check(aclrtSetDevice(0),"device");
  size_t checked=0,launches=0;
  for(int width:{512,1024,6144})for(uint32_t rows:{3u,17u,4096u}) {
    prepared::Context context;
    openpangu::DeviceTensor x({rows,width},ACL_BF16,2),weight({3,width},ACL_BF16,2),
      state({4,2,width},ACL_BF16,2),scratch({rows,width},ACL_BF16,2),
      starts({4},ACL_INT32,4),slots({3},ACL_INT32,4),positions({3},ACL_INT32,4);
    std::vector<uint16_t> w(3*width),history(8*width,bf(.25f));
    for(int j=0;j<3*width;++j)w[j]=bf(float(j%7-3)/32);
    resident::control_upload(weight,w);resident::control_upload(state,history);
    std::vector<uint32_t> prior{0,0,7,1};
    for(int pass=0;pass<3;++pass) {
      // Two single-row requests surround one real extend; reorder state slots
      // on the second pass and preserve a fourth, untouched history.
      std::vector<uint32_t> order=pass==1?std::vector<uint32_t>{0,3,2}:std::vector<uint32_t>{2,0,3};
      std::vector<execution::SequenceSlice> seq;
      for(size_t i=0;i<3;++i)seq.push_back({order[i],prior[order[i]],i==1?rows-2:1,true});
      auto batch=execution::SequenceBatch::make(seq,4,4096,16384);
      std::vector<int32_t> pos;for(auto &s:seq)pos.push_back(s.position);
      std::vector<uint16_t> input(size_t(rows)*width),expected(input.size());
      for(size_t j=0;j<input.size();++j)input[j]=bf(float(int((j*13+pass*17)%101)-50)/64);
      for(size_t s=0;s<seq.size();++s) {
        auto slot=seq[s].resident;
        if(!prior[slot])std::fill(history.begin()+slot*2*width,history.begin()+(slot+1)*2*width,0);
        for(int row=batch.starts[s];row<batch.starts[s+1];++row) {
          for(int j=0;j<width;++j) {
            size_t index=size_t(row)*width+j,base=slot*2*width+j;
            float cur=fp(input[index]),sum=0;
            if(prior[slot]>=2)sum=fp(history[base])*fp(w[j])+fp(history[base+width])*fp(w[width+j])+cur*fp(w[2*width+j]);
            expected[index]=bf(cur+sum);
            history[base]=history[base+width];history[base+width]=input[index];
          }
          ++prior[slot];
        }
      }
      resident::control_upload(x,input);resident::control_upload(starts,batch.starts);
      resident::control_upload(slots,batch.residents);resident::control_upload(positions,pos);context.complete();
      resident::causal_convolution(x.tensor,weight.tensor,state.tensor,starts.tensor,slots.tensor,positions.tensor,scratch.tensor,rows);
      ++launches;
      auto read=[&](openpangu::DeviceTensor &tensor){
        std::vector<uint16_t> result(tensor.bytes/2);void *host=context.output_host(tensor.bytes);
        {resident::TransferScope scope(resident::TransferRole::RequestOutput);
          prepared::check(aclrtMemcpyAsync(host,tensor.bytes,tensor.data,tensor.bytes,ACL_MEMCPY_DEVICE_TO_HOST,context.stream),"fixture output");}
        context.complete();std::memcpy(result.data(),host,tensor.bytes);return result;
      };
      auto actual=read(x);
      for(size_t j=0;j<actual.size();++j) {
        float a=fp(actual[j]),e=fp(expected[j]);
        if(!std::isfinite(a)||std::abs(a-e)>1e-4+.016*std::abs(e))
          throw std::runtime_error("segmented convolution output mismatch width="+std::to_string(width)+" rows="+std::to_string(rows)+" index="+std::to_string(j));
        ++checked;
      }
      if(read(state)!=history)throw std::runtime_error("segmented convolution history isolation mismatch");
    }
  }
  prepared::check(aclrtResetDevice(0),"reset");prepared::check(aclFinalize(),"finalize");
  std::cout<<"{\"passed\":true,\"launches\":"<<launches<<",\"checked_values\":"<<checked<<"}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
