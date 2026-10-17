#pragma once
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

namespace inferfabric::execution {
// Control-only packed sequence layout. Each resident occurs at most once in
// a submission: its recurrent history has one incoming and one outgoing value.
struct SequenceSlice {
  uint32_t resident, position, tokens;
  bool sample;
};
struct SequenceBatch {
  std::vector<int32_t> starts{0}, residents, positions, row_residents;
  // TND attention consumes cumulative query ends and per-sequence KV lengths;
  // convolution consumes each sequence's incoming position instead.
  std::vector<int32_t> query_ends, prior_lengths, kv_lengths;
  std::vector<int64_t> lengths;
  std::vector<uint32_t> sample_rows, sample_residents;

  static SequenceBatch make(const std::vector<SequenceSlice> &sequences,
                            uint32_t resident_capacity, uint32_t token_budget,
                            uint32_t context) {
    if(sequences.empty() || !resident_capacity || !token_budget || !context ||
       resident_capacity>uint32_t(INT32_MAX) || token_budget>uint32_t(INT32_MAX) ||
       context>uint32_t(INT32_MAX))
      throw std::runtime_error("invalid sequence batch limits");
    // Preflight everything before allocating metadata or touching device state.
    uint64_t total=0;
    std::set<uint32_t> seen;
    for(const auto &s:sequences) {
      if(s.resident>=resident_capacity || !seen.insert(s.resident).second ||
         !s.tokens || uint64_t(s.position)+s.tokens>context)
        throw std::runtime_error("invalid sequence batch member");
      total+=s.tokens;
      if(total>token_budget)throw std::runtime_error("sequence batch token budget exceeded");
    }
    SequenceBatch out;
    out.positions.reserve(total);out.row_residents.reserve(total);out.lengths.reserve(total);
    for(const auto &s:sequences) {
      out.residents.push_back(int32_t(s.resident));
      for(uint32_t row=0;row<s.tokens;++row) {
        out.positions.push_back(int32_t(s.position+row));
        out.row_residents.push_back(int32_t(s.resident));
        out.lengths.push_back(int64_t(s.position)+row+1);
      }
      out.starts.push_back(int32_t(out.positions.size()));
      out.query_ends.push_back(int32_t(out.positions.size()));
      out.prior_lengths.push_back(int32_t(s.position));
      out.kv_lengths.push_back(int32_t(s.position+s.tokens));
      if(s.sample) {
        out.sample_rows.push_back(uint32_t(out.positions.size()-1));
        out.sample_residents.push_back(s.resident);
      }
    }
    return out;
  }
};
} // namespace inferfabric::execution
