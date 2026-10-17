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
    if (!s.read(magic, 8) || (std::memcmp(magic, "IFTOKN01", 8) && std::memcmp(magic, "IFTOKN02", 8) && std::memcmp(magic, "IFTOKN03", 8) && std::memcmp(magic, "IFTOKN04", 8) && std::memcmp(magic, "IFTOKN05", 8)))
      throw std::runtime_error("missing/unsupported execution plan");
    Plan p;
    p.shared_pages = !std::memcmp(magic, "IFTOKN05", 8);
    p.physical = std::memcmp(magic, "IFTOKN01", 8) != 0;
    p.context = number();
    p.eos = number();
    p.output = number();
    auto count = number();
    if (!std::memcmp(magic, "IFTOKN03", 8) || !std::memcmp(magic, "IFTOKN04", 8) || p.shared_pages) p.resident_sequences = number();
    if (!std::memcmp(magic, "IFTOKN04", 8) || p.shared_pages) p.prefill_tokens = number();
    if (!p.prefill_tokens || p.prefill_tokens>16 || (p.prefill_tokens&(p.prefill_tokens-1)) || p.prefill_tokens>p.context)
      throw std::runtime_error("prefill token capacity");
    if (!p.resident_sequences || p.resident_sequences > 64)
      throw std::runtime_error("resident sequence capacity");
    if (!p.context || p.context > 512 || p.eos >= 151552 || !count ||
        count > 10000 || p.output >= count)
      throw std::runtime_error("token plan bounds");
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
            (n.kind == 0 ? (n.dtype != "bf16" && n.dtype != "f32") : n.dtype != "opaque"))
          throw std::runtime_error("physical storage contract");
      }
      p.nodes.push_back(std::move(n));
    }
    if (s.peek() != EOF || p.nodes[p.output].kind != 0 ||
        p.nodes[p.output].width != 151552)
      throw std::runtime_error("invalid plan output/trailing data");
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
