#pragma once
#include <cstdint>
#include <climits>
#include <set>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace inferfabric::state {
// Byte ranges within one registered device-state field. The executor must
// validate every field before enqueueing any range, then copy on its own stream.
// These are metadata only: planning never reads cache contents.
struct StateCopyRange {
  uint64_t source, destination, bytes;
};
struct PageSpan { uint64_t offset, bytes; };
// Reset is independent of logical page order. Coalesce physically contiguous
// owned pages only; validate the whole map before any device work is submitted.
inline std::vector<PageSpan> page_spans(const std::vector<int32_t> &pages,
                                      uint64_t page_bytes,uint64_t field_bytes) {
  if(!page_bytes || !field_bytes || field_bytes%page_bytes)
    throw std::runtime_error("page span field layout");
  auto ordered=pages;
  std::sort(ordered.begin(),ordered.end());
  for(size_t i=0;i<ordered.size();++i)
    if(ordered[i]<0 || uint64_t(ordered[i])>=field_bytes/page_bytes || (i && ordered[i]==ordered[i-1]))
      throw std::runtime_error("page span bounds/alias");
  std::vector<PageSpan> spans;
  for(auto page:ordered) {
    const uint64_t offset=uint64_t(page)*page_bytes;
    if(!spans.empty() && spans.back().offset+spans.back().bytes==offset)spans.back().bytes+=page_bytes;
    else spans.push_back({offset,page_bytes});
  }
  return spans;
}
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
// The executable selects fixed partitions or shared exclusive ownership.
// Mapping changes are metadata-only; no resident can name a peer's live pages.
class ResidentPages {
  uint32_t pages_per_slot_, active_ = 0;
  uint32_t page_tokens_, logical_tokens_;
  bool shared_;
  std::vector<PageBinding> slots_;
  std::vector<std::vector<int32_t>> shared_pages_;
  std::vector<int32_t> owners_;
public:
  ResidentPages(uint32_t page_tokens, uint32_t logical_tokens, uint32_t slots, bool shared=false)
      : pages_per_slot_(page_tokens ? logical_tokens / page_tokens : 0),
        page_tokens_(page_tokens),logical_tokens_(logical_tokens),shared_(shared) {
    if (!slots || slots>64 || !page_tokens || !logical_tokens || logical_tokens%page_tokens ||
        uint64_t(pages_per_slot_)*slots>INT32_MAX)
      throw std::runtime_error("resident page capacity");
    for (uint32_t slot=0;slot<slots;++slot) {
      slots_.emplace_back(page_tokens,logical_tokens,pages_per_slot_*slots);
      std::vector<int32_t> pages;
      for(uint32_t i=0;i<pages_per_slot_;++i)pages.push_back(slot*pages_per_slot_+i);
      slots_.back().bind(pages);
      shared_pages_.push_back(pages);
      for(uint32_t i=0;i<pages_per_slot_;++i)owners_.push_back(int32_t(slot));
    }
  }
  void select(uint32_t slot) {
    if(slot>=slots_.size())throw std::runtime_error("resident page slot");
    active_=slot;
  }
  void bind(const std::vector<int32_t> &pages) {
    if(shared_) {
      if(pages.empty() || pages.size()>pages_per_slot_)throw std::runtime_error("shared page binding length");
      std::set<int32_t> seen;
      // Validate the entire replacement before changing ownership.
      for(auto page:pages)
        if(page<0 || size_t(page)>=owners_.size() || !seen.insert(page).second ||
           (owners_[page]!=-1 && owners_[page]!=int32_t(active_)))
          throw std::runtime_error("shared page bounds/owner/alias");
      release(active_);
      shared_pages_[active_]=pages;
      for(auto page:pages)owners_[page]=int32_t(active_);
      return;
    }
    for(auto page:pages)
      if(page<0 || uint32_t(page)/pages_per_slot_!=active_)
        throw std::runtime_error("page binding crosses resident partition");
    slots_[active_].bind(pages);
  }
  void release(uint32_t slot) {
    if(!shared_ || slot>=slots_.size())throw std::runtime_error("shared page release policy/slot");
    for(auto page:shared_pages_[slot])owners_[page]=-1;
    shared_pages_[slot].clear();
  }
  uint64_t append(uint32_t position) const {
    if(!shared_)return slots_[active_].append(position);
    if(position>=logical_tokens_ || position/page_tokens_>=shared_pages_[active_].size())
      throw std::runtime_error("append outside reserved shared pages");
    return uint64_t(shared_pages_[active_][position/page_tokens_])*page_tokens_+position%page_tokens_;
  }
  const std::vector<int32_t> &pages(uint32_t slot) const {
    if(slot>=slots_.size())throw std::runtime_error("page lookup slot");
    return shared_?shared_pages_[slot]:slots_[slot].pages();
  }
  const std::vector<int32_t> &pages() const { return pages(active_); }
  std::vector<StateCopyRange> copy_plan(uint32_t source, uint32_t destination,
                                      uint32_t completed_tokens, bool windowed,
                                      uint64_t field_bytes) const {
    if(source==destination || !completed_tokens || completed_tokens%page_tokens_ ||
       (!windowed && completed_tokens>logical_tokens_))
      throw std::runtime_error("state copy requires distinct slots and a complete page boundary");
    const auto &from=pages(source);
    const auto &to=pages(destination);
    const uint64_t physical_pages=uint64_t(pages_per_slot_)*slots_.size();
    if(!field_bytes || field_bytes%physical_pages)
      throw std::runtime_error("state copy field disagrees with physical page layout");
    const uint64_t page_bytes=field_bytes/physical_pages;
    const size_t count=std::min(completed_tokens/page_tokens_,pages_per_slot_);
    if(from.size()<count || to.size()<count)
      throw std::runtime_error("state copy exceeds reserved source/destination pages");
    std::vector<StateCopyRange> result;
    // Ring ordinals, not chronological order: absolute block tables reconstruct
    // the same modulo mapping at restore. Copy all retained pages after wrap.
    for(size_t i=0;i<count;++i) {
      StateCopyRange next{uint64_t(from[i])*page_bytes,uint64_t(to[i])*page_bytes,page_bytes};
      if(!result.empty() && result.back().source+result.back().bytes==next.source &&
          result.back().destination+result.back().bytes==next.destination)
        result.back().bytes+=page_bytes;
      else result.push_back(next);
    }
    return result;
  }
};
}
