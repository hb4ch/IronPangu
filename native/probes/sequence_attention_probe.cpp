#include "../src/resident_packed_attention.h"
#include "../src/resident_quant_index.h"
#include "../src/page_binding.h"
#include "../include/sequence_batch.h"
#include <iostream>
static float fp(uint16_t b){uint32_t n=uint32_t(b)<<16;float f;std::memcpy(&f,&n,4);return f;}
int main(){using namespace inferfabric;try{
  prepared::check(aclInit(nullptr),"init");prepared::check(aclrtSetDevice(0),"device");
  size_t checked=0,checked_indices=0;
  for(bool learned:{false,true})for(uint32_t rows:{3u,17u,4096u}) {
    prepared::Context context;
    constexpr uint32_t capacity=8192,residents=3,page=32;
    state::ResidentPages pages(page,capacity,residents);
    for(uint32_t slot=0;slot<residents;++slot) {
      pages.select(slot);std::vector<int32_t> ids(capacity/page);
      for(uint32_t i=0;i<ids.size();++i)ids[i]=slot*ids.size()+(i*37)%ids.size();
      pages.bind(ids);
    }
    auto batch=execution::SequenceBatch::make({{2,63,1,true},{0,0,rows-2,false},{1,507,1,true}},3,4096,capacity);
    const int topk=learned?2048:512;
    openpangu::DeviceTensor q({rows,6,576},ACL_BF16,2),cache({residents*capacity/page,page,1,656},ACL_INT8,1),
      indices({rows,1,topk},ACL_INT32,4),tables({residents,capacity/page},ACL_INT32,4),
      ends({residents},ACL_INT32,4),lengths({residents},ACL_INT32,4),sink({128,1,576},ACL_BF16,2),
      sink_values({128,1,512},ACL_BF16,2),output({rows,6,512},ACL_BF16,2);
    std::vector<uint8_t> packed(cache.bytes,0);
    auto value=[](uint32_t slot,uint32_t pos){return int((slot*31+pos*7)%101)-50;};
    const float scale=1.f/128;
    for(uint32_t slot=0;slot<residents;++slot)for(uint32_t pos=0;pos<capacity;++pos) {
      auto *dest=packed.data()+pages.append(slot,pos)*656;
      std::fill(dest,dest+512,uint8_t(int8_t(value(slot,pos))));
      for(int tile=0;tile<4;++tile)std::memcpy(dest+640+tile*4,&scale,4);
    }
    std::vector<int32_t> ids(size_t(rows)*topk,-1);
    for(uint32_t row=0;row<rows;++row)for(int i=0;i<std::min(batch.positions[row]+1,topk);++i)ids[row*topk+i]=i;
    resident::control_upload(q,std::vector<uint16_t>(q.bytes/2,0));
    resident::control_upload(cache,packed);resident::control_upload(indices,ids);
    resident::control_upload(tables,pages.batch_table(batch.residents,capacity/page,false));
    resident::control_upload(ends,batch.query_ends);resident::control_upload(lengths,batch.kv_lengths);
    resident::control_upload(sink,std::vector<uint16_t>(sink.bytes/2,0));
    resident::control_upload(sink_values,std::vector<uint16_t>(sink_values.bytes/2,0));context.complete();
    // Learned index uses independent 64-token pages and its own permutation.
    // Its output is consumed directly by C8 attention without host rewriting.
    state::ResidentPages index_pages(64,capacity,residents);
    openpangu::DeviceTensor iq({rows,24,128},ACL_INT8,1),ik({residents*capacity/64,64,1,128},ACL_INT8,1),
      iw({rows,24},ACL_FLOAT16,2),qs({rows,24},ACL_FLOAT16,2),ks({residents*capacity/64,64,1},ACL_FLOAT16,2),
      itables({residents,capacity/64},ACL_INT32,4);
    auto code=[](int slot,int token){return (token*31337+12345+slot*7919)%65536;};
    auto score=[&](int slot,int token,int row){int c=code(slot,token);
      float dot=float(c%255-127)*std::ldexp(1.f,(c/255)%4-3)*(row%2?-1.f:1.f);
      return std::max(0.f,.5f*dot)+.25f*std::max(0.f,-dot);};
    if(learned) {
      std::vector<int8_t> queries(iq.bytes,0),keys(ik.bytes,0);
      std::vector<uint16_t> ws(iw.bytes/2,0),qsc(qs.bytes/2,0x3c00),ksc(ks.bytes/2);
      for(uint32_t slot=0;slot<residents;++slot) {
        index_pages.select(slot);std::vector<int32_t> blocks(capacity/64);
        for(size_t p=0;p<blocks.size();++p)blocks[p]=slot*blocks.size()+(p*19+7)%blocks.size();
        index_pages.bind(blocks);
        for(uint32_t token=0;token<capacity;++token) {
          auto physical=index_pages.append(slot,token);int c=code(slot,token);
          keys[physical*128]=c%255-127;ksc[physical]=uint16_t(0x3000+((c/255)%4)*0x400);
        }
      }
      for(uint32_t row=0;row<rows;++row) {
        queries[row*24*128]=row%2?-1:1;queries[(row*24+1)*128]=row%2?1:-1;
        ws[row*24]=0x3c00;ws[row*24+1]=0x3400;qsc[row*24]=0x3800;
      }
      resident::control_upload(iq,queries);resident::control_upload(ik,keys);resident::control_upload(iw,ws);
      resident::control_upload(qs,qsc);resident::control_upload(ks,ksc);
      resident::control_upload(itables,index_pages.batch_table(batch.residents,capacity/64,false));
      resident::quant_index(iq.tensor,ik.tensor,iw.tensor,qs.tensor,ks.tensor,ends.tensor,lengths.tensor,itables.tensor,indices.tensor);
    }
    resident::packed_attention(q.tensor,cache.tensor,indices.tensor,tables.tensor,ends.tensor,lengths.tensor,
        sink.tensor,sink_values.tensor,output.tensor);
    void *host=context.output_host(output.bytes);
    {resident::TransferScope scope(resident::TransferRole::RequestOutput);
      prepared::check(aclrtMemcpyAsync(host,output.bytes,output.data,output.bytes,ACL_MEMCPY_DEVICE_TO_HOST,context.stream),"fixture output");}
    context.complete();std::vector<uint16_t> actual(output.bytes/2);std::memcpy(actual.data(),host,output.bytes);
    if(learned) {
      host=context.output_host(indices.bytes);
      {resident::TransferScope scope(resident::TransferRole::RequestOutput);
        prepared::check(aclrtMemcpyAsync(host,indices.bytes,indices.data,indices.bytes,ACL_MEMCPY_DEVICE_TO_HOST,context.stream),"index fixture output");}
      context.complete();std::memcpy(ids.data(),host,indices.bytes);
      for(uint32_t row=0;row<rows;++row) {
        int limit=batch.positions[row]+1,count=std::min(limit,topk),slot=batch.row_residents[row];
        std::vector<float> scores(limit);for(int i=0;i<limit;++i)scores[i]=score(slot,i,row);
        std::nth_element(scores.begin(),scores.begin()+count-1,scores.end(),std::greater<float>());
        std::set<int> seen;
        for(int j=0;j<topk;++j) {
          int id=ids[row*topk+j];
          if(j>=count){if(id!=-1)throw std::runtime_error("batch index padding");continue;}
          if(id<0 || id>=limit || !seen.insert(id).second || score(slot,id,row)<scores[count-1])
            throw std::runtime_error("batch index causal/topk/isolation mismatch row="+std::to_string(row));
          ++checked_indices;
        }
      }
    }
    for(uint32_t row=0;row<rows;++row) {
      int count=std::min(batch.positions[row]+1,topk);double sum=0;
      for(int i=0;i<count;++i)sum+=value(batch.row_residents[row],ids[row*topk+i])*scale;
      double expected=sum/(count+128);
      for(int i=0;i<6*512;++i) {
        double v=fp(actual[size_t(row)*6*512+i]);
        if(!std::isfinite(v)||std::abs(v-expected)>.0001+.02*std::abs(expected))
          throw std::runtime_error("batch attention isolation mismatch rows="+std::to_string(rows)+" row="+std::to_string(row)+" actual="+std::to_string(v)+" expected="+std::to_string(expected));
        ++checked;
      }
    }
  }
  prepared::check(aclrtResetDevice(0),"reset");prepared::check(aclFinalize(),"finalize");
  std::cout<<"{\"passed\":true,\"attention_launches\":6,\"index_launches\":3,\"checked_values\":"<<checked<<",\"checked_indices\":"<<checked_indices<<"}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
