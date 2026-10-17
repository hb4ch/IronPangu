#include "../include/sequence_batch.h"
#include "../src/page_binding.h"
#include <cassert>
#include <iostream>
#include <functional>
using inferfabric::execution::SequenceBatch;
int main() {
  // Decode beside an extend and a final partial chunk. Positions restart per
  // sequence; packed row offsets never become another request's causal length.
  auto b=SequenceBatch::make({{3,100,1,true},{0,4096,4096,false},{2,7,3,true}},4,4100,8192);
  assert((b.starts==std::vector<int32_t>{0,1,4097,4100}));
  assert((b.residents==std::vector<int32_t>{3,0,2}));
  assert((b.query_ends==std::vector<int32_t>{1,4097,4100}));
  assert((b.prior_lengths==std::vector<int32_t>{100,4096,7}));
  assert((b.kv_lengths==std::vector<int32_t>{101,8192,10}));
  assert((b.sample_rows==std::vector<uint32_t>{0,4099}));
  assert((b.sample_residents==std::vector<uint32_t>{3,2}));
  for(size_t i=0;i<b.positions.size();++i) {
    auto slot=i==0?3:i<4097?0:2;
    auto pos=i==0?100:i<4097?4096+i-1:7+i-4097;
    assert(b.row_residents[i]==slot);
    assert(b.positions[i]==int64_t(pos));
    assert(b.lengths[i]==int64_t(pos+1));
  }
  unsigned rejected=0;
  auto reject=[&](std::function<void()> f){try{f();}catch(const std::runtime_error&){++rejected;return;}assert(false);};
  reject([]{SequenceBatch::make({},4,4096,8192);});
  reject([]{SequenceBatch::make({{0,0,1,true},{0,1,1,true}},4,4096,8192);});
  reject([]{SequenceBatch::make({{4,0,1,true}},4,4096,8192);});
  reject([]{SequenceBatch::make({{0,0,0,true}},4,4096,8192);});
  reject([]{SequenceBatch::make({{0,8192,1,true}},4,4096,8192);});
  reject([]{SequenceBatch::make({{0,UINT32_MAX,2,true}},4,4096,8192);});
  reject([]{SequenceBatch::make({{0,0,4096,false},{1,0,1,true}},4,4096,8192);});
  reject([]{SequenceBatch::make({{0,0,1,true}},4,UINT32_MAX,8192);});
  assert(rejected==8);
  for(bool shared:{false,true}) {
    inferfabric::state::ResidentPages pages(16,8192,4,shared);
    // Deliberately select a slot absent from this batch.
    pages.select(1);
    auto selected=pages.append(0);
    for(size_t row=0;row<b.positions.size();++row) {
      auto address=pages.append(uint32_t(b.row_residents[row]),uint32_t(b.positions[row]));
      assert(address==uint64_t(b.row_residents[row])*8192+b.positions[row]);
    }
    assert(pages.append(0)==selected);
    auto tables=pages.batch_table(b.residents,512,false);
    for(size_t seq=0;seq<3;++seq)for(uint32_t page=0;page<512;++page)
      assert(tables[seq*512+page]==b.residents[seq]*512+int32_t(page));
    auto ring=pages.batch_table(b.residents,1025,true);
    for(size_t seq=0;seq<3;++seq)for(uint32_t page=0;page<1025;++page)
      assert(ring[seq*1025+page]==b.residents[seq]*512+int32_t(page%512));
    reject([&]{pages.batch_table({0,0},512,false);});
    reject([&]{pages.batch_table({-1},512,false);});
    reject([&]{pages.batch_table({0},511,false);});
    reject([&]{pages.append(4,0);});
    if(shared) {
      pages.release(2);
      reject([&]{pages.append(2,7);});
      assert(pages.append(0)==selected);
    }
  }
  // Legacy single-sequence lowering must retain every position/length exactly.
  for(uint32_t count:{1u,2u,17u,4096u})for(uint32_t position:{0u,511u,4096u}) {
    auto single=SequenceBatch::make({{0,position,count,false}},1,4096,8192);
    for(uint32_t i=0;i<count;++i) {
      assert(single.positions[i]==int64_t(position+i));
      assert(single.lengths[i]==int64_t(position+i+1));
    }
    assert((single.starts==std::vector<int32_t>{0,int32_t(count)}));
  }
  std::cout<<"sequence batch: 4100 mixed rows isolated; explicit fixed/shared addressing; "
           <<rejected<<" invalid submissions rejected; 12 single-sequence equivalence cases\n";
}
