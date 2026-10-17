#include "openpangu_probe_common.h"
#include "../src/openpangu_ops.h"
#include "../include/sliding_window_layout.h"
#include "../src/openpangu_decoder_state.h"
#include <aclnnop/aclnn_index_copy.h>
#include <aclnn_ai_infra_fused_infer_attention_sink.h>
#include <aclnn_ai_infra_fused_infer_attention_sink_metadata.h>
using openpangu::DeviceTensor;
using openpangu::Operation;
void production_state_scatter() {
  pangu::AttentionState state(16,544,false,2,true,65536);
  state.page_binding.release(0);state.page_binding.release(1);
  std::vector<int32_t> owned(34);for(int i=0;i<34;++i)owned[i]=67-i;
  state.page_binding.bind(owned);
  auto blocks=state.blocks();state.table.upload(blocks);
  for(uint32_t pos=0;pos<65536;++pos) {
    auto expected=uint64_t(owned[(pos/16)%34])*16+pos%16;
    if(state.append(pos)!=expected || blocks[pos/16]!=int32_t(expected/16))
      throw std::runtime_error("production ring table/append disagreement");
  }
  state.page_binding.select(1);
  bool rejected=false;try{state.page_binding.bind({owned[0]});}catch(const std::runtime_error&){rejected=true;}
  if(!rejected)throw std::runtime_error("production ring cross-owner alias");
  state.page_binding.select(0);
  std::vector<uint16_t> expected(state.key.bytes/2,0);state.key.upload(expected);
  DeviceTensor source({4,576},ACL_BF16,2),index({4},ACL_INT32,4);
  int64_t shape[]={1088,576},stride[]={576,1};
  auto cache=aclCreateTensor(shape,2,ACL_BF16,stride,0,ACL_FORMAT_ND,shape,2,state.key.data);
  if(!cache)throw std::runtime_error("scatter cache descriptor");
  aclrtStream stream=nullptr;aclmdlRI graph=nullptr;check(aclrtCreateStream(&stream),"scatter stream");
  Operation op;
  auto setup=[&](uint32_t start) {
    std::vector<int32_t> indices(4);std::vector<uint16_t> rows(4*576);
    for(uint32_t r=0;r<4;++r) {
      indices[r]=int32_t(state.append(start+r));
      for(int d=0;d<576;++d) {
        auto value=bf16(float(int(((start+r)*17+d*13)%101)-50)/128);
        rows[r*576+d]=value;expected[size_t(indices[r])*576+d]=value;
      }
    }
    source.upload(rows);index.upload(indices);
  };
  auto prepare=[&]{check(aclnnInplaceIndexCopyGetWorkspaceSize(cache,0,index.tensor,source.tensor,&op.bytes,&op.executor),"ring scatter prepare");};
  setup(0);prepare();op.allocate();
  check(aclmdlRICaptureBegin(stream,ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),"scatter capture");
  op.run(aclnnInplaceIndexCopy,stream);check(aclmdlRICaptureEnd(stream,&graph),"scatter capture end");
  std::vector<uint32_t> starts;
  for(uint32_t start=0;start<2048;start+=4)starts.push_back(start);
  for(uint32_t start=32760;start<32784;start+=4)starts.push_back(start);
  for(uint32_t start=65512;start<65536;start+=4)starts.push_back(start);
  for(auto start:starts) {
    setup(start);check(aclmdlRIExecuteAsync(graph,stream),"scatter replay");
    check(aclrtSynchronizeStream(stream),"scatter completion");
  }
  if(state.key.read<uint16_t>()!=expected)throw std::runtime_error("production ring scatter mismatch or peer overwritten");
  state.page_binding.release(0);state.page_binding.select(1);state.page_binding.bind(owned);
  if(state.blocks()!=blocks)throw std::runtime_error("production ring reassignment differs");
  check(aclmdlRIDestroy(graph),"scatter graph destroy");check(aclrtDestroyStream(stream),"scatter stream destroy");aclDestroyTensor(cache);
  std::cout<<"{\"production_state_scatter\":\"passed\",\"checked_addresses\":65536,\"graph_replays\":"<<starts.size()<<",\"peer_storage_preserved\":true}\n";
}
// Qualification only: synthetic fixtures and output readback are not serving fallbacks.
template<int Rows,int Window=512> struct WindowProbe {
  static constexpr int Chunk=Rows>4?Rows:4;
  inferfabric::state::SlidingWindowLayout layout{65536,Window,Chunk,16};
  aclrtStream stream=nullptr;aclmdlRI graph=nullptr;
  DeviceTensor q{{Rows,6,512},ACL_BF16,2},qr{{Rows,6,64},ACL_BF16,2};
  DeviceTensor k,v,kr,table;
  DeviceTensor sink{{128,1,512},ACL_BF16,2},sr{{128,1,64},ACL_BF16,2};
  DeviceTensor qlen{{1},ACL_INT64,8},klen{{1},ACL_INT64,8};
  DeviceTensor mask{{2048,2048},ACL_BOOL,1},meta{{1024},ACL_INT32,4},out{{Rows,6,512},ACL_BF16,2};
  aclTensorList *keys=nullptr,*values=nullptr;
  Operation metadata,attention;
  int64_t cube=0,vector=0;size_t checked_values=0,replays=0;double max_error=0;
  std::vector<uint16_t> hq,hqr,hs,hsr;
  static uint16_t key(int position,int d) {return bf16(float((int64_t(position)*17+d*13)%101-50)/128);}
  explicit WindowProbe():k({layout.physical_pages,16,512},ACL_BF16,2),
      v({layout.physical_pages,16,512},ACL_BF16,2),kr({layout.physical_pages,16,64},ACL_BF16,2),
      table({1,layout.table_entries()},ACL_INT32,4) {
    check(aclrtCreateStream(&stream),"stream");
    check(aclrtGetDeviceInfo(0,ACL_DEV_ATTR_CUBE_CORE_NUM,&cube),"cube cores");
    check(aclrtGetDeviceInfo(0,ACL_DEV_ATTR_VECTOR_CORE_NUM,&vector),"vector cores");
    // Tensor lists own their descriptors. Do not hand them DeviceTensor's
    // descriptor: that would be destroyed twice when several probes coexist.
    auto cache_list=[&](DeviceTensor &cache) {
      int64_t shape[]={layout.physical_pages,16,512},stride[]={16*512,512,1};
      auto *view=aclCreateTensor(shape,3,ACL_BF16,stride,0,ACL_FORMAT_ND,shape,3,cache.data);
      if(!view)throw std::runtime_error("cache list descriptor");
      auto *list=aclCreateTensorList(&view,1);
      if(!list)aclDestroyTensor(view);
      return list;
    };
    keys=cache_list(k);values=cache_list(v);
    if(!keys || !values)throw std::runtime_error("tensor lists");
    std::vector<int32_t> blocks(layout.table_entries());
    for(uint32_t i=0;i<blocks.size();++i)blocks[i]=layout.physical_page(i);
    table.upload(blocks);qlen.upload(std::vector<int64_t>{Rows});
    std::vector<uint8_t> hm(2048*2048,0);
    for(int i=0;i<2048;++i)for(int j=i+1;j<2048;++j)hm[i*2048+j]=1;
    mask.upload(hm);meta.upload(std::vector<int32_t>(1024,0));
    hq.resize(Rows*6*512);hqr.resize(Rows*6*64);hs.resize(128*512);hsr.resize(128*64);
    for(size_t i=0;i<hq.size();++i)hq[i]=bf16(float(int(i*7%97)-48)/128);
    for(size_t i=0;i<hqr.size();++i)hqr[i]=bf16(float(int(i*11%89)-44)/128);
    for(size_t i=0;i<hs.size();++i)hs[i]=bf16(float(int(i*19%103)-51)/128);
    for(size_t i=0;i<hsr.size();++i)hsr[i]=bf16(float(int(i*23%107)-53)/128);
    q.upload(hq);qr.upload(hqr);sink.upload(hs);sr.upload(hsr);
    setup(0,Rows,Rows-1);prepare_metadata();metadata.allocate();prepare_attention();attention.allocate();
  }
  void prepare_metadata() {
    char l[]="TND",kv[]="BnBsH";
    check(aclnnAiInfraFusedInferAttentionSinkMetadataGetWorkspaceSize(qlen.tensor,klen.tensor,
      6,1,512,512,1,4,Window-1,0,l,kv,0,128,64,16,cube,vector,meta.tensor,
      &metadata.bytes,&metadata.executor),"window metadata prepare");
  }
  void prepare_attention() {
    char l[]="TND";
    check(aclnnAiInfraFusedInferAttentionSinkGetWorkspaceSize(q.tensor,keys,values,
      nullptr,mask.tensor,qlen.tensor,klen.tensor,
      nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
      table.tensor,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
      qr.tensor,kr.tensor,nullptr,nullptr,meta.tensor,sink.tensor,sr.tensor,sink.tensor,
      6,1/std::sqrt(192.),Window-1,0,l,1,4,0,16,0,false,0,0,0,0,out.tensor,nullptr,
      &attention.bytes,&attention.executor),"window attention prepare");
  }
  void launch() {
    if(!metadata.executor)prepare_metadata();
    metadata.run(aclnnAiInfraFusedInferAttentionSinkMetadata,stream);
    if(!attention.executor)prepare_attention();
    attention.run(aclnnAiInfraFusedInferAttentionSink,stream);
  }
  void setup(int start,int count,int row) {
    // Materialize only the retained union, with all chunk writes already done.
    // Fill other storage with poison so accidentally reading stale pages fails.
    std::vector<uint16_t> hk(k.bytes/2,bf16(4)),hr(kr.bytes/2,bf16(4));
    std::vector<int> owner(layout.physical_pages*16,-1);
    for(int pos=std::max(0,start-Window+1);pos<start+count;++pos) {
      auto slot=layout.append(pos);
      if(owner[slot]!=-1)throw std::runtime_error("chunk overwrote live window");
      owner[slot]=pos;
      for(int d=0;d<512;++d)hk[slot*512+d]=key(pos,d);
      for(int d=0;d<64;++d)hr[slot*64+d]=key(pos,512+d);
    }
    k.upload(hk);v.upload(hk);kr.upload(hr);klen.upload(std::vector<int64_t>{start+row+1});
  }
  void verify(int end) {
    auto actual=out.read<uint16_t>();
    for(int row=0;row<Rows;++row){int pos=end-Rows+1+row,first=std::max(0,pos-Window+1),n=pos-first+1;
    for(int h=0;h<6;++h) {
      std::vector<double> scores(n+128);double maximum=-1e30;
      for(int j=0;j<n+128;++j) {
        double s=0;
        for(int d=0;d<512;++d)s+=double(fp32(hq[(row*6+h)*512+d]))*fp32(j<n?key(first+j,d):hs[(j-n)*512+d]);
        for(int d=0;d<64;++d)s+=double(fp32(hqr[(row*6+h)*64+d]))*fp32(j<n?key(first+j,512+d):hsr[(j-n)*64+d]);
        scores[j]=s/std::sqrt(192.);maximum=std::max(maximum,scores[j]);
      }
      double total=0;for(auto &s:scores){s=std::exp(s-maximum);total+=s;}
      for(int d=0;d<512;++d) {
        double expected=0;for(int j=0;j<n+128;++j)expected+=scores[j]/total*fp32(j<n?key(first+j,d):hs[(j-n)*512+d]);
        double value=fp32(actual[(row*6+h)*512+d]),error=std::abs(value-expected);max_error=std::max(max_error,error);
        if(!std::isfinite(value) || error>0.001+0.02*std::abs(expected))
          throw std::runtime_error("window CPU mismatch position="+std::to_string(pos)+" error="+std::to_string(error));
        ++checked_values;
      }
    }}
  }
  void run() {
    std::vector<int> starts{0,15,16,509,511,512,525,527,528,1023,2047,4095,16383,32767,65536-Chunk};
    starts.push_back(layout.physical_pages*16-1);
    starts.push_back(layout.physical_pages*16);
    std::vector<std::vector<uint16_t>> eager;
    for(int start:starts)for(int row=0;row<(Rows==1?4:1);++row) {
      int last=Rows==1?row:Rows-1;
      setup(start,Chunk,last);launch();check(aclrtSynchronizeStream(stream),"eager fence");verify(start+last);eager.push_back(out.read<uint16_t>());
    }
    setup(0,Chunk,Rows-1);check(aclmdlRICaptureBegin(stream,ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),"capture");launch();
    check(aclmdlRICaptureEnd(stream,&graph),"capture end");
    for(int pass=0;pass<2;++pass) {size_t i=0;for(int start:starts)for(int row=0;row<(Rows==1?4:1);++row) {
      int last=Rows==1?row:Rows-1;
      setup(start,Chunk,last);check(aclmdlRIExecuteAsync(graph,stream),"replay");check(aclrtSynchronizeStream(stream),"replay fence");verify(start+last);
      if(out.read<uint16_t>()!=eager[i++])throw std::runtime_error("window graph differs from eager");
      ++replays;
    }}
  }
  ~WindowProbe(){if(stream)aclrtSynchronizeStream(stream);if(graph)aclmdlRIDestroy(graph);
    if(keys)aclDestroyTensorList(keys);
    if(values)aclDestroyTensorList(values);
    if(stream)aclrtDestroyStream(stream);}
};
template<int Rows,int Window=512> void qualify(){
  WindowProbe<Rows,Window> p;p.run();std::cout<<"{\"status\":\"passed\",\"query_rows\":"<<Rows<<",\"max_context\":65536,\"window\":"<<Window<<",\"chunk\":"<<p.Chunk<<",\"physical_pages\":"<<p.layout.physical_pages
    <<",\"checked_values\":"<<p.checked_values<<",\"graph_replays\":"<<p.replays<<",\"max_abs_error\":"<<p.max_error<<"}\n";
}
int main(int argc,char **argv){try{
bool all=argc==2 && std::string(argv[1])=="all";
int rows=argc==2 && !all?std::stoi(argv[1]):1;
if(argc>2 || (rows!=1 && rows!=4 && rows!=16))throw std::runtime_error("query count must be 1,4,16");
check(aclInit(nullptr),"init");check(aclrtSetDevice(0),"device");{
  if(rows==1){production_state_scatter();qualify<1>();}
  else if(rows==4)qualify<4>();else qualify<16>();
  if(all){qualify<4>();qualify<16>();qualify<4,2048>();}
}check(aclrtResetDevice(0),"reset");check(aclFinalize(),"finalize");}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
