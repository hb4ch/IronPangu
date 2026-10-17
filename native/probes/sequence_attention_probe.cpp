#include "../src/resident_packed_attention.h"
#include "../src/page_binding.h"
#include "../include/sequence_batch.h"
#include <iostream>
static float fp(uint16_t b){uint32_t n=uint32_t(b)<<16;float f;std::memcpy(&f,&n,4);return f;}
int main(){using namespace inferfabric;try{
  prepared::check(aclInit(nullptr),"init");prepared::check(aclrtSetDevice(0),"device");
  size_t checked=0;
  for(uint32_t rows:{3u,17u,4096u}) {
    prepared::Context context;
    constexpr uint32_t capacity=8192,residents=3,page=32;
    state::ResidentPages pages(page,capacity,residents);
    for(uint32_t slot=0;slot<residents;++slot) {
      pages.select(slot);std::vector<int32_t> ids(capacity/page);
      for(uint32_t i=0;i<ids.size();++i)ids[i]=slot*ids.size()+(i*37)%ids.size();
      pages.bind(ids);
    }
    auto batch=execution::SequenceBatch::make({{2,63,1,true},{0,0,rows-2,false},{1,507,1,true}},3,4096,capacity);
    openpangu::DeviceTensor q({rows,6,576},ACL_BF16,2),cache({residents*capacity/page,page,1,656},ACL_INT8,1),
      indices({rows,1,512},ACL_INT32,4),tables({residents,capacity/page},ACL_INT32,4),
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
    std::vector<int32_t> ids(size_t(rows)*512,-1);
    for(uint32_t row=0;row<rows;++row)for(int i=0;i<std::min(batch.positions[row]+1,512);++i)ids[row*512+i]=i;
    resident::control_upload(q,std::vector<uint16_t>(q.bytes/2,0));
    resident::control_upload(cache,packed);resident::control_upload(indices,ids);
    resident::control_upload(tables,pages.batch_table(batch.residents,capacity/page,false));
    resident::control_upload(ends,batch.query_ends);resident::control_upload(lengths,batch.kv_lengths);
    resident::control_upload(sink,std::vector<uint16_t>(sink.bytes/2,0));
    resident::control_upload(sink_values,std::vector<uint16_t>(sink_values.bytes/2,0));context.complete();
    resident::packed_attention(q.tensor,cache.tensor,indices.tensor,tables.tensor,ends.tensor,lengths.tensor,
        sink.tensor,sink_values.tensor,output.tensor);
    void *host=context.output_host(output.bytes);
    {resident::TransferScope scope(resident::TransferRole::RequestOutput);
      prepared::check(aclrtMemcpyAsync(host,output.bytes,output.data,output.bytes,ACL_MEMCPY_DEVICE_TO_HOST,context.stream),"fixture output");}
    context.complete();auto *actual=static_cast<uint16_t *>(host);
    for(uint32_t row=0;row<rows;++row) {
      int count=std::min(batch.positions[row]+1,512);double sum=0;
      for(int i=0;i<count;++i)sum+=value(batch.row_residents[row],i)*scale;
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
  std::cout<<"{\"passed\":true,\"launches\":3,\"checked_values\":"<<checked<<"}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
