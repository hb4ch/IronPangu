#pragma once
// Model-independent token program reader. Kernel semantics live in the backend;
// this file contains no model names, layer counts, or weight naming convention.
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
namespace inferfabric::execution {
struct Node {
  std::string dtype, kernel;
  uint32_t logical_node = 0;
  uint32_t kind = 0, width = 0;
  std::string name, op;
  std::vector<uint32_t> inputs;
  std::map<std::string, std::string> attrs;
  const std::string &text(const std::string &key) const {
    return attrs.at(key);
  }
  uint32_t integer(const std::string &key) const {
    const auto &s = text(key);
    size_t end = 0;
    auto n = std::stoull(s, &end);
    if (end != s.size() || n > 20000000 || s[0] == '-')
      throw std::runtime_error("invalid plan integer");
    return static_cast<uint32_t>(n);
  }
};
struct Plan {
  bool physical = false;
  uint32_t context = 0, eos = 0, output = 0;
  uint32_t resident_sequences = 1;
  uint32_t prefill_tokens = 1;
  bool prefill_family = false;
  bool uncaptured_prefill = false;
  std::vector<uint32_t> prefill_variants;
  bool supports_prefill(uint32_t count) const {
    if(uncaptured_prefill)return count>1 && count<=prefill_tokens;
    for(auto variant:prefill_variants)if(variant==count)return true;
    return false;
  }
  bool bulk_collective_domain = false;
  bool compact_output_rows = false;
  uint32_t output_row_source=0,output_row_width=0,output_row_first=0,output_row_last=0;
  bool shared_pages = false;
  std::vector<Node> nodes;
  std::vector<uint32_t> uses;
  static Plan read(const std::string &path) {
    std::ifstream s(path, std::ios::binary);
    auto number = [&]() {
      uint32_t v = 0;
      for (int i = 0; i < 4; ++i) {
        int c = s.get();
        if (c == EOF)
          throw std::runtime_error("truncated token plan");
        v |= uint32_t(uint8_t(c)) << (8 * i);
      }
      return v;
    };
    auto str = [&]() {
      auto n = number();
      if (!n || n > 1024)
        throw std::runtime_error("token plan string bound");
      std::string v(n, '\0');
      if (!s.read(v.data(), n) || v.find('\0') != std::string::npos)
        throw std::runtime_error("invalid token plan string");
      return v;
    };
    char magic[8];
    if (!s.read(magic, 8) || (std::memcmp(magic, "IFTOKN01", 8) && std::memcmp(magic, "IFTOKN02", 8) && std::memcmp(magic, "IFTOKN03", 8) && std::memcmp(magic, "IFTOKN04", 8) && std::memcmp(magic, "IFTOKN05", 8) && std::memcmp(magic, "IFTOKN06", 8) && std::memcmp(magic, "IFTOKN07", 8) && std::memcmp(magic, "IFTOKN08", 8) && std::memcmp(magic, "IFTOKN09", 8)))
      throw std::runtime_error("missing/unsupported execution plan");
    Plan p;
    p.bulk_collective_domain = !std::memcmp(magic, "IFTOKN09", 8);
    p.compact_output_rows = !std::memcmp(magic, "IFTOKN08", 8);
    p.uncaptured_prefill = !std::memcmp(magic, "IFTOKN07", 8) || p.compact_output_rows || p.bulk_collective_domain;
    p.prefill_family = !std::memcmp(magic, "IFTOKN06", 8);
    p.shared_pages = !std::memcmp(magic, "IFTOKN05", 8);
    p.physical = std::memcmp(magic, "IFTOKN01", 8) != 0;
    p.context = number();
    p.eos = number();
    p.output = number();
    auto count = number();
    if (!std::memcmp(magic, "IFTOKN03", 8) || !std::memcmp(magic, "IFTOKN04", 8) || p.shared_pages || p.prefill_family || p.uncaptured_prefill) p.resident_sequences = number();
    if (!std::memcmp(magic, "IFTOKN04", 8) || p.shared_pages || p.prefill_family || p.uncaptured_prefill) p.prefill_tokens = number();
    if (!p.prefill_tokens || (p.uncaptured_prefill ? p.prefill_tokens!=4096 : p.prefill_tokens>32) || (p.prefill_tokens&(p.prefill_tokens-1)) || p.prefill_tokens>p.context)
      throw std::runtime_error("prefill token capacity");
    if(p.uncaptured_prefill) {
      auto shared=number();
      if(shared>1)throw std::runtime_error("uncaptured prefill shared-page flag");
      p.shared_pages=shared;
    } else if(p.prefill_family) {
      auto shared=number(), variants=number();
      if(shared>1 || !variants || variants>5)throw std::runtime_error("prefill family header");
      p.shared_pages=shared;
      uint32_t previous=1;
      for(uint32_t i=0;i<variants;++i) {
        auto n=number();
        if(n<=previous || n>p.prefill_tokens || (n&(n-1)))throw std::runtime_error("exact prefill variant bounds/order");
        p.prefill_variants.push_back(n);previous=n;
      }
      if(previous!=p.prefill_tokens)throw std::runtime_error("prefill family maximum");
    } else if(p.prefill_tokens>1) p.prefill_variants.push_back(p.prefill_tokens);
    if (!p.resident_sequences || p.resident_sequences > 64)
      throw std::runtime_error("resident sequence capacity");
    if (!p.context || p.context > 65536 || p.eos >= 151552 || !count ||
        count > 10000 || p.output >= count)
      throw std::runtime_error("token plan bounds");
    if(p.bulk_collective_domain) {
      auto compact=number();
      if(compact>1)throw std::runtime_error("bulk domain output-row flag");
      p.compact_output_rows=compact;
    }
    if(p.compact_output_rows) {
      p.output_row_source=number();p.output_row_width=number();p.output_row_first=number();p.output_row_last=number();
      if(p.output_row_source>=p.output_row_first || p.output_row_first>=count || p.output_row_last!=p.output || p.output!=count-1)
        throw std::runtime_error("compact output range" );
    }
    p.uses.resize(count);
    std::set<std::string> names;
    for (uint32_t i = 0; i < count; ++i) {
      Node n;
      n.kind = number();
      n.width = number();
      n.name = str();
      n.op = str();
      if ((p.resident_sequences > 1 || p.prefill_tokens > 1 || p.shared_pages) && n.op == "cache_quantize_append")
        throw std::runtime_error("resident slots require paged cache operators");
      if (n.kind > (p.physical ? 9u : 4u) || (n.width > 200000 && !(p.physical && n.op=="cache_dequantize" && n.width==((p.context+15)/16)*16*576)) || !names.insert(n.name).second)
        throw std::runtime_error("invalid node descriptor");
      auto inputs = number();
      if (inputs > 8)
        throw std::runtime_error("node input bound");
      for (uint32_t j = 0; j < inputs; ++j) {
        auto id = number();
        if (id >= i)
          throw std::runtime_error("non-topological plan");
        n.inputs.push_back(id);
        ++p.uses[id];
      }
      auto attrs = number();
      if (attrs > 16)
        throw std::runtime_error("node attribute bound");
      for (uint32_t j = 0; j < attrs; ++j) {
        auto k = str();
        auto v = str();
        if (!n.attrs.emplace(k, v).second)
          throw std::runtime_error("duplicate attribute");
      }
      if (p.physical) {
        n.dtype = str(); n.kernel = str(); n.logical_node = number();
        if (n.logical_node >= count ||
            (n.kind == 0 ? (n.dtype != "bf16" && n.dtype != "f16" && n.dtype != "f32") : n.dtype != "opaque"))
          throw std::runtime_error("physical storage contract");
      }
      p.nodes.push_back(std::move(n));
    }
    if (s.peek() != EOF || p.nodes[p.output].kind != 0 ||
        p.nodes[p.output].width != 151552)
      throw std::runtime_error("invalid plan output/trailing data");
    if(p.compact_output_rows) {
      if(p.nodes[p.output_row_source].kind!=0 || p.nodes[p.output_row_source].width!=p.output_row_width || !p.output_row_width)
        throw std::runtime_error("compact output source");
      for(uint32_t i=p.output_row_first;i<=p.output_row_last;++i) {
        const auto &n=p.nodes[i];
        if(n.kind!=0 || n.inputs.size()!=1 || n.inputs[0]!=(i==p.output_row_first?p.output_row_source:i-1))
          throw std::runtime_error("compact output dependency");
        bool ordinary=n.op=="norm" || n.op=="linear" || n.op=="gather" || n.op=="cast" || n.op=="copy";
        bool pair=n.op=="gather_heads" && i<p.output_row_last && p.nodes[i+1].op=="head_to_token" &&
          n.width==p.nodes[i+1].width && n.attrs.count("heads") && p.nodes[i+1].attrs.count("heads") &&
          n.text("heads")==p.nodes[i+1].text("heads");
        bool layout=n.op=="head_to_token" && i>p.output_row_first && p.nodes[i-1].op=="gather_heads";
        if(!ordinary && !pair && !layout)throw std::runtime_error("unsafe compact output operation");
      }
    }
    ++p.uses[p.output];
    std::set<uint32_t> live{p.output};
    for (uint32_t i = count; i-- > 0;)
      if (live.count(i))
        live.insert(p.nodes[i].inputs.begin(), p.nodes[i].inputs.end());
    if (live.size() != count)
      throw std::runtime_error("unreachable plan node");
    return p;
  }
};
} // namespace inferfabric::execution
