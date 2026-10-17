#include "../src/resident_window_attention.h"
#include "../src/page_binding.h"
#include "../include/sequence_batch.h"
#include "../include/sliding_window_layout.h"
#include <iostream>
static uint16_t bf(float v){uint32_t n;std::memcpy(&n,&v,4);return uint16_t((n+0x7fff+((n>>16)&1))>>16);}
static float fp(uint16_t v){uint32_t n=uint32_t(v)<<16;float f;std::memcpy(&f,&n,4);return f;}
struct CacheList {
  aclTensorList *value;
  explicit CacheList(openpangu::DeviceTensor &tensor) {
    int64_t shape[]={int64_t(tensor.bytes/(16*512*2)),16,512},strides[]={8192,512,1};
    auto *view=aclCreateTensor(shape,3,ACL_BF16,strides,0,ACL_FORMAT_ND,shape,3,tensor.data);
    if(!view)throw std::runtime_error("cache view");
    value=aclCreateTensorList(&view,1);
    if(!value){aclDestroyTensor(view);throw std::runtime_error("cache list");}
  }
  ~CacheList(){aclDestroyTensorList(value);}
};
int main(){using namespace inferfabric;try{
  prepared::check(aclInit(nullptr),"init");prepared::check(aclrtSetDevice(0),"device");
  int64_t cube=0,vector=0;
  prepared::check(aclrtGetDeviceInfo(0,ACL_DEV_ATTR_CUBE_CORE_NUM,&cube),"cube");
  prepared::check(aclrtGetDeviceInfo(0,ACL_DEV_ATTR_VECTOR_CORE_NUM,&vector),"vector");
  size_t checked=0;
  for(uint32_t window:{128u,512u})for(uint32_t rows:{3u,17u,4096u}) {
    prepared::Context context;
    state::SlidingWindowLayout layout(65536,window,rows,16);
    state::ResidentPages pages(16,layout.physical_pages*16,3);
    for(uint32_t slot=0;slot<3;++slot) {
      pages.select(slot);std::vector<int32_t> ids(layout.physical_pages);
      for(size_t i=0;i<ids.size();++i)ids[i]=(slot+1)*ids.size()-1-i;
      pages.bind(ids);
    }
    auto batch=execution::SequenceBatch::make({{2,65535,1,true},{0,32767,rows-2,false},{1,0,1,true}},3,4096,65536);
    const uint32_t physical=layout.physical_pages*3,ring=layout.physical_pages*16;
    openpangu::DeviceTensor q({rows,6,512},ACL_BF16,2),qr({rows,6,64},ACL_BF16,2),
      k({physical,16,512},ACL_BF16,2),kr({physical,16,64},ACL_BF16,2),
      tables({3,layout.table_entries()},ACL_INT32,4),ends({3},ACL_INT64,8),lengths({3},ACL_INT64,8),
      sink({128,1,512},ACL_BF16,2),sr({128,1,64},ACL_BF16,2),
      mask({2048,2048},ACL_BOOL,1),metadata({1024},ACL_INT32,4),out({rows,6,512},ACL_BF16,2);
    std::vector<uint16_t> cache(k.bytes/2,bf(7)),rope(kr.bytes/2,0);
    auto value=[](uint32_t slot,uint32_t pos){return float(int((slot*31+uint64_t(pos)*7)%101)-50)/128;};
    for(size_t seq=0;seq<batch.residents.size();++seq) {
      int first=std::max(0,batch.prior_lengths[seq]-int(window)+1);
      for(int pos=first;pos<batch.kv_lengths[seq];++pos) {
        auto physical_row=pages.append(batch.residents[seq],pos%ring);
        std::fill(cache.begin()+physical_row*512,cache.begin()+(physical_row+1)*512,bf(value(batch.residents[seq],pos)));
      }
    }
    resident::control_upload(k,cache);resident::control_upload(kr,rope);
    resident::control_upload(q,std::vector<uint16_t>(q.bytes/2,0));resident::control_upload(qr,std::vector<uint16_t>(qr.bytes/2,0));
    resident::control_upload(sink,std::vector<uint16_t>(sink.bytes/2,0));resident::control_upload(sr,std::vector<uint16_t>(sr.bytes/2,0));
    resident::control_upload(tables,pages.batch_table(batch.residents,layout.table_entries(),true));
    resident::control_upload(ends,std::vector<int64_t>(batch.query_ends.begin(),batch.query_ends.end()));
    resident::control_upload(lengths,std::vector<int64_t>(batch.kv_lengths.begin(),batch.kv_lengths.end()));
    std::vector<uint8_t> triangle(mask.bytes,0);for(int i=0;i<2048;++i)for(int j=i+1;j<2048;++j)triangle[i*2048+j]=1;
    resident::control_upload(mask,triangle);context.zero(metadata.data,metadata.bytes);context.complete();
    CacheList keys(k),values(k);
    resident::window_metadata(ends.tensor,lengths.tensor,window,cube,vector,metadata.tensor);
    resident::window_attention(q.tensor,keys.value,values.value,mask.tensor,ends.tensor,lengths.tensor,tables.tensor,
        qr.tensor,kr.tensor,metadata.tensor,sink.tensor,sr.tensor,window,out.tensor);
    void *host=context.output_host(out.bytes);
    {resident::TransferScope scope(resident::TransferRole::RequestOutput);
      prepared::check(aclrtMemcpyAsync(host,out.bytes,out.data,out.bytes,ACL_MEMCPY_DEVICE_TO_HOST,context.stream),"fixture output");}
    context.complete();auto *actual=static_cast<uint16_t *>(host);
    for(uint32_t row=0;row<rows;++row) {
      int pos=batch.positions[row],first=std::max(0,pos-int(window)+1);double sum=0;
      for(int token=first;token<=pos;++token)sum+=value(batch.row_residents[row],token);
      double expected=sum/(pos-first+1+128);
      for(int i=0;i<6*512;++i) {
        double v=fp(actual[size_t(row)*3072+i]);
        if(!std::isfinite(v)||std::abs(v-expected)>.0001+.02*std::abs(expected))
          throw std::runtime_error("window batch mismatch window="+std::to_string(window)+" rows="+std::to_string(rows)+" row="+std::to_string(row)+" actual="+std::to_string(v)+" expected="+std::to_string(expected));
        ++checked;
      }
    }
  }
  prepared::check(aclFinalize(),"finalize");
  std::cout<<"{\"passed\":true,\"batches\":6,\"checked_values\":"<<checked<<"}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
