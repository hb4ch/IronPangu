#include "../src/page_binding.h"
#include <iostream>
#include <numeric>
#include <cstring>
using inferfabric::state::page_spans;
void require(bool ok){if(!ok)throw std::runtime_error("page span test failed");}
template<class F>void rejects(F f){try{f();}catch(const std::runtime_error&){return;}throw std::runtime_error("invalid page map accepted");}
void compare(const std::vector<int32_t> &pages,uint64_t page_bytes,uint64_t total_pages) {
  std::vector<uint8_t> expected(total_pages*page_bytes,0xA7),actual=expected;
  for(auto page:pages)std::memset(expected.data()+uint64_t(page)*page_bytes,0,page_bytes);
  uint64_t bytes=0;
  for(const auto &span:page_spans(pages,page_bytes,actual.size())) {
    require(span.offset+span.bytes<=actual.size());
    std::memset(actual.data()+span.offset,0,span.bytes);bytes+=span.bytes;
  }
  require(actual==expected && bytes==pages.size()*page_bytes);
}
int main(){
  for(uint64_t bytes:{2u,128u,656u,1152u}) {
    compare({},bytes,10);compare({4,0,3,1,8},bytes,10);
    compare({0,2,4,6,8},bytes,10);compare({9},bytes,10);
  }
  std::vector<int32_t> pages(2048);std::iota(pages.begin(),pages.end(),2048);
  std::reverse(pages.begin(),pages.end());
  compare(pages,32*656,6144); // one 64K C8 resident among three
  auto spans=page_spans(pages,32*656,uint64_t(6144)*32*656);
  require(spans.size()==1 && spans[0].offset==uint64_t(2048)*32*656 && spans[0].bytes==uint64_t(2048)*32*656);
  rejects([]{page_spans({-1},16,160);});rejects([]{page_spans({10},16,160);});
  rejects([]{page_spans({1,1},16,160);});rejects([]{page_spans({1},0,160);});
  rejects([]{page_spans({1},16,159);});rejects([]{page_spans({},16,0);});
  std::cout<<"page reset spans: sparse ownership, byte equivalence and 64K coalescing passed\n";
}
