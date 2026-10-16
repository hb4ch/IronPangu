#pragma once
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace inferfabric::prepared {
inline void check(int status, const char *operation) {
  if (status)
    throw std::runtime_error(std::string(operation) + ": " +
                             std::to_string(status));
}
class Context;
inline thread_local Context *active = nullptr;

// A lease may be recycled after its last use has been ENQUEUED on this stream.
// No other stream may touch these buffers. Physical memory and descriptors stay
// alive until the context's completion boundary, including on error unwinding.
class Context {
  struct Block {
    void *data;
    size_t bytes;
    bool live = true;
  };
  struct Invocation {
    aclOpExecutor *executor;
    uint64_t bytes;
  };
  struct Counts {
    uint64_t preparations = 0, replays = 0;
    bool repeatable = false;
  };
  std::vector<std::unique_ptr<Block>> blocks_;
  std::map<size_t, std::vector<Block *>> free_;
  std::map<void *, Block *> owned_;
  std::map<std::string, aclTensor *> descriptors_;
  std::map<std::string, Invocation> invocations_;
  std::map<std::string, Counts> counts_;
  struct Graph {
    aclmdlRI handle;
    std::string name;
    uint64_t replays = 0;
  };
  std::vector<Graph> graphs_;
  bool capturing_ = false;
  std::vector<std::pair<void *, size_t>> uploads_;
  size_t upload_index_ = 0;
  void *workspace_ = nullptr;
  size_t workspace_bytes_ = 0;
  aclScalar *one_ = nullptr;
  void *output_host_ = nullptr;
  size_t output_capacity_ = 0;
  uint64_t allocations_ = 0, reuse_ = 0, allocated_bytes_ = 0, boundaries_ = 0;
  struct Step {
    int position;
    uint64_t allocations, reuses, boundaries;
  };
  std::vector<Step> steps_;
  Step before_{};

public:
  aclrtStream stream = nullptr;
  Context() {
    if (active)
      throw std::runtime_error("nested prepared execution context");
    check(aclrtCreateStream(&stream), "prepared stream");
    float one = 1;
    one_ = aclCreateScalar(&one, ACL_FLOAT);
    if (!one_)
      throw std::runtime_error("prepared scalar");
    active = this;
  }
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  ~Context() {
    // A stream-local teardown boundary, never aclrtSynchronizeDevice.
    aclrtSynchronizeStream(stream);
    release_graphs();
    active = nullptr;
    for (auto &[key, invocation] : invocations_)
      aclDestroyAclOpExecutor(invocation.executor);
    for (auto &[key, tensor] : descriptors_)
      aclDestroyTensor(tensor);
    aclDestroyScalar(one_);
    for (auto &block : blocks_)
      aclrtFree(block->data);
    for (auto &[pointer, bytes] : uploads_)
      aclrtFreeHost(pointer);
    if (output_host_)
      aclrtFreeHost(output_host_);
    aclrtDestroyStream(stream);
  }
  void require_stream(aclrtStream candidate) const {
    if (candidate != stream || active != this)
      throw std::runtime_error(
          "prepared buffer used outside its owning stream/context");
  }
  void *acquire(size_t bytes) {
    require_stream(stream);
    if (!bytes)
      throw std::runtime_error("zero prepared allocation");
    auto &available = free_[bytes];
    if (!available.empty()) {
      auto *block = available.back();
      available.pop_back();
      block->live = true;
      ++reuse_;
      return block->data;
    }
    if (capturing_)
      throw std::runtime_error("capture requires prepared device buffers");
    void *pointer = nullptr;
    check(aclrtMalloc(&pointer, bytes, ACL_MEM_MALLOC_HUGE_FIRST),
          "prepared allocation");
    auto block = std::make_unique<Block>(Block{pointer, bytes});
    owned_.emplace(pointer, block.get());
    blocks_.push_back(std::move(block));
    ++allocations_;
    allocated_bytes_ += bytes;
    return pointer;
  }
  void release(void *pointer) {
    require_stream(stream);
    auto found = owned_.find(pointer);
    if (found == owned_.end() || !found->second->live)
      throw std::runtime_error("invalid prepared buffer release");
    auto *block = found->second;
    block->live = false;
    free_[block->bytes].push_back(block);
  }
  template <class T> static void key_part(std::string &key, const T &value) {
    static_assert(std::is_trivially_copyable_v<T>);
    key.append(reinterpret_cast<const char *>(&value), sizeof(value));
  }
  static void key_part(std::string &key, const char *value) {
    key.append(value ? value : "");
    key.push_back('\0');
  }
  static void key_part(std::string &key, char *value) {
    key_part(key, const_cast<const char *>(value));
  }
  aclTensor *descriptor(void *pointer, const std::vector<int64_t> &shape,
                        aclDataType dtype, bool transpose = false) {
    require_stream(stream);
    if (shape.empty())
      throw std::runtime_error("empty prepared shape");
    std::string key;
    key_part(key, pointer);
    key_part(key, dtype);
    key_part(key, transpose);
    for (auto d : shape)
      key_part(key, d);
    auto found = descriptors_.find(key);
    if (found != descriptors_.end())
      return found->second;
    auto dimensions = shape;
    std::vector<int64_t> strides(shape.size(), 1);
    for (size_t j = shape.size() - 1; j > 0; --j)
      strides[j - 1] = strides[j] * shape[j];
    if (transpose) {
      std::swap(dimensions.at(0), dimensions.at(1));
      std::swap(strides.at(0), strides.at(1));
    }
    auto *tensor = aclCreateTensor(dimensions.data(), dimensions.size(), dtype,
                                   strides.data(), 0, ACL_FORMAT_ND,
                                   shape.data(), shape.size(), pointer);
    if (!tensor)
      throw std::runtime_error("prepared tensor descriptor");
    descriptors_.emplace(std::move(key), tensor);
    return tensor;
  }
  aclScalar *one() const { return one_; }
  void *workspace(size_t bytes) {
    if (bytes > workspace_bytes_) {
      // The previous workspace remains allocated; subsequent reuse is ordered
      // after all its previous users on the same stream.
      if (workspace_)
        release(workspace_);
      workspace_ = acquire(bytes);
      workspace_bytes_ = bytes;
    }
    return workspace_;
  }
  size_t workspace_capacity() const { return workspace_bytes_; }
  void copy(void *dst, const void *src, size_t bytes) {
    require_stream(stream);
    check(aclrtMemcpyAsync(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE,
                           stream),
          "prepared D2D");
  }
  void upload(void *dst, const void *src, size_t bytes) {
    require_stream(stream);
    // Pinned staging is immutable until complete(); never retain the caller's
    // temporary vector. Each queued upload has its own slot.
    if (upload_index_ == uploads_.size())
      uploads_.emplace_back(nullptr, 0);
    auto &[pointer, capacity] = uploads_[upload_index_++];
    if (capacity < bytes) {
      if (pointer)
        aclrtFreeHost(pointer);
      check(aclrtMallocHost(&pointer, bytes), "prepared pinned input");
      capacity = bytes;
    }
    std::memcpy(pointer, src, bytes);
    check(aclrtMemcpyAsync(dst, bytes, pointer, bytes,
                           ACL_MEMCPY_HOST_TO_DEVICE, stream),
          "prepared H2D");
  }
  void zero(void *dst, size_t bytes) {
    check(aclrtMemsetAsync(dst, bytes, 0, bytes, stream), "prepared zero");
  }
  void complete() {
    check(aclrtSynchronizeStream(stream), "prepared request-output completion");
    ++boundaries_;
    upload_index_ = 0;
  }
  template <class F> aclmdlRI capture(const char *name, F enqueue) {
    require_stream(stream);
    if (capturing_)
      throw std::runtime_error("nested stream capture");
    // Preparation only; no global device/rank barrier and no replay fence.
    check(aclrtSynchronizeStream(stream), "capture input preparation");
    check(aclmdlRICaptureBegin(stream, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),
          "capture begin");
    capturing_ = true;
    try {
      enqueue();
    } catch (...) {
      aclmdlRI failed = nullptr;
      aclmdlRICaptureEnd(stream, &failed);
      capturing_ = false;
      if (failed)
        aclmdlRIDestroy(failed);
      throw;
    }
    aclmdlRI graph = nullptr;
    auto status = aclmdlRICaptureEnd(stream, &graph);
    capturing_ = false;
    check(status, "capture end");
    graphs_.push_back({graph, name, 0});
    return graph;
  }
  void replay(aclmdlRI handle) {
    require_stream(stream);
    for (auto &graph : graphs_)
      if (graph.handle == handle) {
        check(aclmdlRIExecuteAsync(handle, stream), "graph replay");
        ++graph.replays;
        return;
      }
    throw std::runtime_error("graph belongs to a different execution context");
  }
  void release_graphs() {
    // Caller completes the owning stream before resource teardown.
    for (auto &graph : graphs_)
      aclmdlRIDestroy(graph.handle);
    graphs_.clear();
  }
  void *output_host(size_t bytes) {
    // Only used at the serialized request-output boundary; previous output
    // copies have completed before the caller starts the next token.
    if (bytes > output_capacity_) {
      if (output_host_)
        aclrtFreeHost(output_host_);
      check(aclrtMallocHost(&output_host_, bytes), "prepared pinned output");
      output_capacity_ = bytes;
    }
    return output_host_;
  }
  void begin_step(int position) {
    before_ = {position, allocations_, reuse_, boundaries_};
  }
  void end_step() {
    steps_.push_back({before_.position, allocations_ - before_.allocations,
                      reuse_ - before_.reuses,
                      boundaries_ - before_.boundaries});
  }
  template <class F, class Tuple, size_t... I>
  int query(const char *name, bool repeatable, F function, Tuple &args,
            std::index_sequence<I...>) {
    auto *bytes = std::get<sizeof...(I)>(args);
    auto **executor = std::get<sizeof...(I) + 1>(args);
    auto &counts = counts_[name];
    counts.repeatable = repeatable;
    if (!repeatable) {
      ++counts.preparations;
      return std::apply(function, args);
    }
    std::string key(name);
    key_part(key, function);
    (key_part(key, std::get<I>(args)), ...);
    auto found = invocations_.find(key);
    if (found != invocations_.end()) {
      *bytes = found->second.bytes;
      *executor = found->second.executor;
      ++counts.replays;
      return 0;
    }
    int status = std::apply(function, args);
    if (status)
      return status;
    check(aclSetAclOpExecutorRepeatable(*executor), name);
    invocations_.emplace(std::move(key), Invocation{*executor, *bytes});
    ++counts.preparations;
    return 0;
  }
  void write(std::ostream &out) const {
    out << "{\"format\":\"inferfabric.prepared-execution.v1\",\"ordering\":"
           "\"single-stream\","
        << "\"device_allocations\":" << allocations_
        << ",\"device_bytes\":" << allocated_bytes_
        << ",\"buffer_reuses\":" << reuse_
        << ",\"descriptors\":" << descriptors_.size()
        << ",\"output_boundaries\":" << boundaries_ << ",\"invocations\":[";
    bool first = true;
    for (const auto &[name, counts] : counts_) {
      if (!first)
        out << ',';
      first = false;
      out << "{\"name\":\"" << name
          << "\",\"repeatable\":" << (counts.repeatable ? "true" : "false")
          << ",\"preparations\":" << counts.preparations
          << ",\"replays\":" << counts.replays << '}';
    }
    out << "],\"steps\":[";
    first = true;
    for (const auto &step : steps_) {
      if (!first)
        out << ',';
      first = false;
      out << "{\"position\":" << step.position
          << ",\"device_allocations\":" << step.allocations
          << ",\"buffer_reuses\":" << step.reuses
          << ",\"completion_boundaries\":" << step.boundaries << '}';
    }
    out << "],\"graphs\":[";
    first = true;
    for (const auto &graph : graphs_) {
      if (!first)
        out << ',';
      first = false;
      out << "{\"name\":\"" << graph.name << "\",\"replays\":" << graph.replays
          << '}';
    }
    out << "]}\n";
  }
};
template <class F, class... Args>
int query(const char *name, bool repeatable, F function, Args... values) {
  auto args = std::make_tuple(values...);
  if (!active)
    return std::apply(function, args);
  return active->query(name, repeatable, function, args,
                       std::make_index_sequence<sizeof...(Args) - 2>{});
}
} // namespace inferfabric::prepared
