#include "../src/page_binding.h"
#include <cstring>
#include <iostream>
#include <numeric>

using inferfabric::state::ResidentPages;
void require(bool ok) { if(!ok)throw std::runtime_error("state copy assertion failed"); }
template<class F> void rejects(F fn) {
  bool failed=false;try { fn(); }catch(const std::runtime_error &) {failed=true;}
  require(failed);
}

// Host byte arrays are an oracle for address planning only. Production state
// will execute these ranges as D2D copies; no tensor readback API is introduced.
void check(uint32_t page,uint32_t capacity,uint32_t completed,bool ring,uint32_t row_bytes) {
  const uint32_t count=capacity/page;
  ResidentPages binding(page,capacity,3,true);
  for(uint32_t slot=0;slot<3;++slot)binding.release(slot);
  std::vector<int32_t> source(count),destination(count);
  for(uint32_t i=0;i<count;++i){source[i]=3*i+2;destination[i]=3*(count-1-i);}
  binding.bind(source);binding.select(1);binding.bind(destination);binding.select(0);
  std::vector<uint8_t> storage(uint64_t(capacity)*3*row_bytes,0xAB);
  for(uint32_t token=0;token<completed;++token) {
    const auto offset=binding.append(ring?token%capacity:token)*row_bytes;
    for(uint32_t column=0;column<row_bytes;++column)
      storage[offset+column]=uint8_t((token*17+column*31)%251);
  }
  auto before=storage;
  const auto plan=binding.copy_plan(0,1,completed,ring,storage.size());
  uint64_t bytes=0;
  for(const auto &range:plan) {
    require(range.source+range.bytes<=storage.size() && range.destination+range.bytes<=storage.size());
    std::memcpy(storage.data()+range.destination,storage.data()+range.source,range.bytes);
    bytes+=range.bytes;
  }
  require(bytes==uint64_t(std::min(capacity,completed))*row_bytes);
  binding.select(1);
  const auto first=completed>capacity?completed-capacity:0;
  for(uint32_t token=first;token<completed;++token) {
    const auto offset=binding.append(ring?token%capacity:token)*row_bytes;
    for(uint32_t column=0;column<row_bytes;++column)
      require(storage[offset+column]==uint8_t((token*17+column*31)%251));
  }
  for(auto physical:source)
    for(uint64_t i=uint64_t(physical)*page*row_bytes;i<uint64_t(physical+1)*page*row_bytes;++i)
      require(before[i]==storage[i]);
  // A divergent continuation must not change the immutable source checkpoint.
  const auto next=binding.append(ring?completed%capacity:0)*row_bytes;
  storage[next]=0;
  for(auto physical:source)
    require(before[uint64_t(physical)*page*row_bytes]==storage[uint64_t(physical)*page*row_bytes]);
}
int main() {
  for(uint32_t boundary:{64u,512u,576u,32768u,65536u}) {
    check(16,544,boundary,true,1152); // retained BF16 window
    check(32,65536,boundary,false,656); // packed C8 incl rotary and scales
    check(64,65536,boundary,false,128); // index keys
    check(64,65536,boundary,false,2); // separate index scales, same ownership
  }
  ResidentPages p(32,128,3,true);
  auto merged=p.copy_plan(0,1,128,false,384);
  require(merged.size()==1 && merged[0].source==0 && merged[0].destination==128 && merged[0].bytes==128);
  rejects([&]{p.copy_plan(0,0,64,false,384);});
  rejects([&]{p.copy_plan(0,3,64,false,384);});
  rejects([&]{p.copy_plan(0,1,0,false,384);});
  rejects([&]{p.copy_plan(0,1,63,false,384);});
  rejects([&]{p.copy_plan(0,1,160,false,384);});
  rejects([&]{p.copy_plan(0,1,64,false,383);});
  p.release(1);p.select(1);p.bind({4});
  rejects([&]{p.copy_plan(0,1,64,false,384);});
  require(p.pages(0)==std::vector<int32_t>({0,1,2,3}));
  require(p.pages(1)==std::vector<int32_t>({4}));
  std::cout<<"state-copy planning: 64K fragmented pages, ring restore, independent scales, isolation and rejection passed\n";
}
