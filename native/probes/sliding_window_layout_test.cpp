#include "../include/sliding_window_layout.h"
#include <algorithm>
#include <iostream>
#include <vector>
using inferfabric::state::SlidingWindowLayout;
template<class F> void rejects(F f) {
  bool rejected=false;try{f();}catch(const std::runtime_error&){rejected=true;}
  if(!rejected)throw std::runtime_error("invalid layout accepted");
}
int main() {
  uint64_t chunks=0;
  for(uint32_t width:{1u,4u,16u,32u}) {
    SlidingWindowLayout layout(65536,512,width,16);
    std::vector<int> owners(layout.physical_pages*16,-1);
    for(uint32_t start=0;start+width<=layout.context;++start) {
      std::fill(owners.begin(),owners.end(),-1);
      auto first=start<511?0:start-511;
      for(auto p=first;p<start+width;++p) {
        auto slot=layout.append(p);
        if(owners[slot]!=-1)throw std::runtime_error("live token overwritten");
        owners[slot]=int(p);
        if(slot/16!=layout.physical_page(p/16))throw std::runtime_error("table and write disagree");
      }
      ++chunks;
    }
    rejects([&]{layout.append(65536);});
    rejects([&]{layout.physical_page(layout.table_entries());});
  }
  for(uint32_t context:{1u,15u,16u,17u,129u,512u}) {
    SlidingWindowLayout layout(context,512,1,16);
    if(layout.physical_pages!=(context+15)/16)throw std::runtime_error("short context overallocated");
  }
  rejects([]{SlidingWindowLayout(0,512,4,16);});
  rejects([]{SlidingWindowLayout(65536,0,4,16);});
  rejects([]{SlidingWindowLayout(65536,512,0,16);});
  rejects([]{SlidingWindowLayout(65536,512,4,0);});
  rejects([]{SlidingWindowLayout(1,512,4,16);});
  rejects([]{SlidingWindowLayout(UINT32_MAX,UINT32_MAX,1,16);});
  std::cout<<"{\"status\":\"passed\",\"exhaustive_chunk_starts\":"<<chunks<<"}\n";
}
