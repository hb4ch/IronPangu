#include "openpangu_probe_common.h"
#include "../src/openpangu_ops.h"
#include <aclnn_ai_infra_sparse_flash_attention_pioneer.h>
#include <limits>
#include "../src/openpangu_attention_metadata.h"
using openpangu::DeviceTensor; using openpangu::Operation;
// Fixed addresses, mutable paged metadata: two sequences and four packed queries.
struct Attention {
 int heads; aclrtStream stream=nullptr; aclmdlRI graph=nullptr;
 DeviceTensor q,k,v,indices,table,qlen,klen,sink,sv,out; Operation op;
 std::vector<uint16_t> hostq,hostk,hosts; std::vector<int32_t> selected,blocks,lengths,ends;
 size_t checked_values=0; double max_error=0;
 explicit Attention(int h):heads(h),q({4,h,576},ACL_BF16,2),k({8,16,1,576},ACL_BF16,2),v({8,16,1,512},ACL_BF16,2),indices({4,1,16},ACL_INT32,4),table({2,4},ACL_INT32,4),qlen({2},ACL_INT32,4),klen({2},ACL_INT32,4),sink({128,1,576},ACL_BF16,2),sv({128,1,512},ACL_BF16,2),out({4,h,512},ACL_BF16,2) {
  check(aclrtCreateStream(&stream),"attention stream"); setup(0); query(); op.allocate();
 }
 void query(){char a[]="TND",b[]="PA_BSND";check(aclnnAiInfraSparseFlashAttentionPioneerGetWorkspaceSize(q.tensor,k.tensor,v.tensor,indices.tensor,table.tensor,qlen.tensor,klen.tensor,nullptr,nullptr,sink.tensor,sv.tensor,1/std::sqrt(192.0),1,a,b,3,INT64_MAX,INT64_MAX,2,false,out.tensor,nullptr,nullptr,&op.bytes,&op.executor),"sparse attention prepare");}
 void launch(){if(!op.executor)query();op.run(aclnnAiInfraSparseFlashAttentionPioneer,stream);}
 void setup(int variant){
  hostq.resize(4*heads*576);hostk.resize(128*576);hosts.resize(128*576);
  for(size_t i=0;i<hostq.size();++i)hostq[i]=bf16(float(int((i*13+variant*17)%101)-50)/128);
  for(size_t i=0;i<hostk.size();++i)hostk[i]=bf16(float(int((i*7+variant*11)%97)-48)/128);
  for(size_t i=0;i<hosts.size();++i)hosts[i]=bf16(float(int((i*19)%103)-51)/128);
  blocks=variant%2?std::vector<int32_t>{6,1,4,3,2,7,0,5}:std::vector<int32_t>{3,0,7,2,5,1,6,4};
  lengths=variant==0?std::vector<int32_t>{35,19}:std::vector<int32_t>{17,49};if(variant==3)lengths={2,3};ends={2,4};selected.assign(64,-1);
  for(int t=0;t<4;++t){int limit=lengths[t/2]-2+t%2+1,pos=0;for(int j=0;j<std::min(16,limit);++j)selected[t*16+pos++]=variant==2?j*(limit-1)/15:j;}
  openpangu::validate_sparse_attention_indices(ends,lengths,selected,16);q.upload(hostq);k.upload(hostk);sink.upload(hosts);std::vector<uint16_t> values(128*512),sinkvalues(128*512);for(int row=0;row<128;++row)for(int d=0;d<512;++d){values[row*512+d]=hostk[row*576+d];sinkvalues[row*512+d]=hosts[row*576+d];}v.upload(values);sv.upload(sinkvalues);indices.upload(selected);table.upload(blocks);qlen.upload(ends);klen.upload(lengths);
 }
 void verify(){auto actual=out.read<uint16_t>();for(int t=0;t<4;++t)for(int h=0;h<heads;++h){std::vector<const uint16_t*> keys;for(int j=0;j<16;++j){int id=selected[t*16+j];if(id>=0)keys.push_back(hostk.data()+(blocks[t/2*4+id/16]*16+id%16)*576);}for(int j=0;j<128;++j)keys.push_back(hosts.data()+j*576);
  std::vector<double> scores;double maximum=-1e30;for(auto key:keys){double s=0;for(int d=0;d<576;++d)s+=double(fp32(hostq[(t*heads+h)*576+d]))*fp32(key[d]);s/=std::sqrt(192.0);scores.push_back(s);maximum=std::max(maximum,s);}double total=0;for(auto &s:scores){s=std::exp(s-maximum);total+=s;}
  for(int d=0;d<512;++d){double expected=0;for(size_t j=0;j<keys.size();++j)expected+=scores[j]/total*fp32(keys[j][d]);double value=fp32(actual[(t*heads+h)*512+d]),error=std::abs(value-expected);max_error=std::max(max_error,error);if(!std::isfinite(value)||error>0.001+0.02*std::abs(expected))throw std::runtime_error("attention CPU reference mismatch t="+std::to_string(t)+" h="+std::to_string(h)+" d="+std::to_string(d)+" actual="+std::to_string(value)+" expected="+std::to_string(expected));++checked_values;}
 }}
 void run(){auto invalid=selected;invalid[0]=-1;bool rejected=false;try{openpangu::validate_sparse_attention_indices(ends,lengths,invalid,16);}catch(const std::runtime_error&){rejected=true;}if(!rejected)throw std::runtime_error("underfilled sparse candidates accepted");std::vector<std::vector<uint16_t>> eager;for(int n=0;n<4;++n){setup(n);launch();check(aclrtSynchronizeStream(stream),"attention fence");verify();eager.push_back(out.read<uint16_t>());}setup(0);check(aclmdlRICaptureBegin(stream,ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),"attention capture");launch();check(aclmdlRICaptureEnd(stream,&graph),"attention capture end");for(int pass=0;pass<2;++pass)for(int n=0;n<4;++n){setup(n);check(aclmdlRIExecuteAsync(graph,stream),"attention replay");check(aclrtSynchronizeStream(stream),"attention replay fence");verify();if(out.read<uint16_t>()!=eager[n])throw std::runtime_error("attention graph mismatch");}}
 ~Attention(){if(stream)aclrtSynchronizeStream(stream);if(graph)aclmdlRIDestroy(graph);if(stream)aclrtDestroyStream(stream);}
};
int main(int argc,char**argv){try{int heads=argc==2?std::stoi(argv[1]):6;if(heads!=6&&heads!=48)throw std::runtime_error("heads must be 6 or 48");check(aclInit(nullptr),"init");check(aclrtSetDevice(0),"device");{Attention a(heads);a.run();std::cout<<"{\"status\":\"passed\",\"heads\":"<<heads<<",\"checked_values\":"<<a.checked_values<<",\"max_abs_error\":"<<a.max_error<<"}\n";}check(aclrtResetDevice(0),"reset");check(aclFinalize(),"finalize");}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
