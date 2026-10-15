#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <vector>
namespace openpangu {
// Validate host scheduling metadata before uploading to the captured operator.
// Pioneer expects min(causal_length, topk) entries, followed by -1 padding.
inline void validate_sparse_attention_indices(
    const std::vector<int32_t>& ends, const std::vector<int32_t>& lengths,
    const std::vector<int32_t>& indices, int topk) {
  if (ends.empty() || ends.size()!=lengths.size() || topk<=0 || ends.back()<=0 ||
      uint64_t(ends.back())*uint64_t(topk)!=indices.size())
    throw std::runtime_error("invalid sparse attention metadata dimensions");
  int start=0;
  for (size_t sequence=0;sequence<ends.size();++sequence) {
    int count=ends[sequence]-start;
    if(count<0 || lengths[sequence]<count)
      throw std::runtime_error("invalid sparse attention sequence length");
    for(int t=start;t<ends[sequence];++t) {
      int causal=lengths[sequence]-count+t-start+1;
      int required=std::min(causal,topk);
      std::unordered_set<int32_t> seen;
      for(int j=0;j<topk;++j) {
        int id=indices[size_t(t)*topk+j];
        if(j<required) {
          if(id<0 || id>=causal || !seen.insert(id).second)
            throw std::runtime_error("sparse attention needs distinct causal candidates without holes");
        } else if(id!=-1) {
          throw std::runtime_error("sparse attention padding must be -1");
        }
      }
    }
    start=ends[sequence];
  }
}
} // namespace openpangu

