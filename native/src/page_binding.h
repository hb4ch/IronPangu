#pragma once
#include <cstdint>
#include <climits>
#include <set>
#include <stdexcept>
#include <vector>

namespace inferfabric::state {
// Host metadata for a stable device block table. Cache rows never visit the host.
class PageBinding {
  uint32_t page_tokens_, logical_tokens_, physical_pages_;
  std::vector<int32_t> pages_;
public:
  PageBinding(uint32_t page_tokens, uint32_t logical_tokens, uint32_t physical_pages)
      : page_tokens_(page_tokens), logical_tokens_(logical_tokens), physical_pages_(physical_pages) {
    if (!page_tokens || !logical_tokens || physical_pages>INT32_MAX || logical_tokens % page_tokens ||
        logical_tokens / page_tokens > physical_pages)
      throw std::runtime_error("invalid page binding capacity");
    for (uint32_t i=0;i<logical_tokens/page_tokens;++i) pages_.push_back(int32_t(i));
  }
  void bind(const std::vector<int32_t> &pages) {
    if (pages.size()!=pages_.size()) throw std::runtime_error("page binding length");
    std::set<int32_t> seen;
    for (auto page : pages)
      if (page<0 || uint32_t(page)>=physical_pages_ || !seen.insert(page).second)
        throw std::runtime_error("page binding bounds/alias");
    pages_=pages;
  }
  uint64_t append(uint32_t position) const {
    if (position>=logical_tokens_) throw std::runtime_error("page append bounds");
    return uint64_t(pages_[position/page_tokens_])*page_tokens_+position%page_tokens_;
  }
  const std::vector<int32_t> &pages() const { return pages_; }
};
// Fixed disjoint partitions are the first shared-pool allocation policy. Mapping
// changes are metadata-only; no resident can name another resident's pages.
class ResidentPages {
  uint32_t pages_per_slot_, active_ = 0;
  std::vector<PageBinding> slots_;
public:
  ResidentPages(uint32_t page_tokens, uint32_t logical_tokens, uint32_t slots)
      : pages_per_slot_(page_tokens ? logical_tokens / page_tokens : 0) {
    if (!slots || slots>64 || !page_tokens || !logical_tokens || logical_tokens%page_tokens ||
        uint64_t(pages_per_slot_)*slots>INT32_MAX)
      throw std::runtime_error("resident page capacity");
    for (uint32_t slot=0;slot<slots;++slot) {
      slots_.emplace_back(page_tokens,logical_tokens,pages_per_slot_*slots);
      std::vector<int32_t> pages;
      for(uint32_t i=0;i<pages_per_slot_;++i)pages.push_back(slot*pages_per_slot_+i);
      slots_.back().bind(pages);
    }
  }
  void select(uint32_t slot) {
    if(slot>=slots_.size())throw std::runtime_error("resident page slot");
    active_=slot;
  }
  void bind(const std::vector<int32_t> &pages) {
    for(auto page:pages)
      if(page<0 || uint32_t(page)/pages_per_slot_!=active_)
        throw std::runtime_error("page binding crosses resident partition");
    slots_[active_].bind(pages);
  }
  uint64_t append(uint32_t position) const { return slots_[active_].append(position); }
  const std::vector<int32_t> &pages() const { return slots_[active_].pages(); }
};
}
