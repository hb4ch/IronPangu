#include "../src/page_binding.h"
#include <cassert>
#include <iostream>
template<class F> void rejects(F f) {
  bool rejected=false; try { f(); } catch(const std::runtime_error &) { rejected=true; }
  if(!rejected)throw std::runtime_error("invalid page binding accepted");
}
int main() {
  using inferfabric::state::PageBinding;
  for(auto page_tokens:{16u,32u}) {
    PageBinding p(page_tokens,page_tokens*3,7);
    p.bind({6,0,4});
    assert(p.append(0)==6*page_tokens);
    assert(p.append(page_tokens-1)==7*page_tokens-1);
    assert(p.append(page_tokens)==0);
    assert(p.append(3*page_tokens-1)==5*page_tokens-1);
    rejects([&]{ p.append(3*page_tokens); });
    rejects([&]{ p.bind({0,0,1}); });
    rejects([&]{ p.bind({0,-1,1}); });
    rejects([&]{ p.bind({0,7,1}); });
    rejects([&]{ p.bind({0,1}); });
    assert(p.append(0)==6*page_tokens); // failed update must preserve old mapping
  }
  rejects([]{ PageBinding p(0,32,2); });
  rejects([]{ PageBinding p(16,31,2); });
  rejects([]{ PageBinding p(16,64,3); });
  rejects([]{ PageBinding p(16,16,UINT32_MAX); });
  for(auto page_tokens:{16u,32u}) {
    inferfabric::state::ResidentPages p(page_tokens,page_tokens*3,2);
    p.bind({2,0,1});
    p.select(1);
    assert(p.append(0)==3*page_tokens);
    p.bind({5,3,4});
    rejects([&]{p.bind({0,3,4});});
    rejects([&]{p.select(2);});
    assert(p.append(0)==5*page_tokens);
    p.select(0);
    assert(p.append(0)==2*page_tokens);
    rejects([&]{p.bind({3,0,1});});
    assert(p.append(0)==2*page_tokens);
  }
  std::cout<<"page bindings: fragmented addresses and invalid-update rollback passed\n";
}
