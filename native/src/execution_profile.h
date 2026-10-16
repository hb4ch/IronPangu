#pragma once
#include <acl/acl.h>
#include <chrono>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
namespace inferfabric::profiling {
struct Stat {
  uint64_t calls = 0, ns = 0;
};
struct State {
  bool active = false;
  std::map<std::string, Stat> nodes, apis;
};
inline thread_local State state;
struct Scope {
  bool enabled;
  bool node;
  std::string name;
  std::chrono::steady_clock::time_point start;
  explicit Scope(const char *label, bool logical = false)
      : enabled(state.active), node(logical) {
    if (enabled) {
      name = label;
      start = std::chrono::steady_clock::now();
    }
  }
  ~Scope() {
    if (enabled) {
      auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start)
                    .count();
      auto &s = (node ? state.nodes : state.apis)[name];
      ++s.calls;
      s.ns += ns;
    }
  }
};
inline void write(const std::string &path) {
  std::ofstream out(path);
  out << "kind,name,calls,host_duration_ns\n";
  for (const auto &entry : state.nodes)
    out << "node," << entry.first << ',' << entry.second.calls << ','
        << entry.second.ns << '\n';
  for (const auto &entry : state.apis)
    out << "api," << entry.first << ',' << entry.second.calls << ','
        << entry.second.ns << '\n';
  if (!out)
    throw std::runtime_error("profile write failed");
}
inline aclError malloc(void **ptr, size_t bytes, aclrtMemMallocPolicy policy) {
  Scope s("aclrtMalloc");
  return aclrtMalloc(ptr, bytes, policy);
}
inline aclError free(void *ptr) {
  Scope s("aclrtFree");
  return aclrtFree(ptr);
}
inline aclError synchronize(aclrtStream stream) {
  Scope s("aclrtSynchronizeStream");
  return aclrtSynchronizeStream(stream);
}
} // namespace inferfabric::profiling
#define aclrtMalloc inferfabric::profiling::malloc
#define aclrtFree inferfabric::profiling::free
#define aclrtSynchronizeStream inferfabric::profiling::synchronize
