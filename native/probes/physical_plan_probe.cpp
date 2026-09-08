// Hardware qualification adapter for verified planner fixtures, not a serving
// ABI.
#include <acl/acl.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_relu.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
void check(int code, const char *what) {
  if (code)
    throw std::runtime_error(std::string(what) + ": " + std::to_string(code));
}
struct Buffer {
  int kind;
  int64_t offset;
  std::vector<int64_t> shape;
  std::vector<float> initial;
  size_t bytes = 4;
  void *ptr = nullptr;
  aclTensor *tensor = nullptr;
};
using Launch = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
struct Step {
  int op;
  size_t a, b, y;
  uint64_t bytes = 0;
  aclOpExecutor *executor = nullptr;
  Launch launch = nullptr;
};
struct Binding {
  size_t id;
  std::vector<float> values;
};
struct Case {
  std::vector<Binding> inputs, expected;
};
struct Runner {
  std::ifstream in;
  std::vector<Buffer> buffers;
  std::vector<Step> steps;
  std::vector<std::pair<size_t, size_t>> updates;
  std::vector<void *> snapshots, allocations;
  std::vector<Case> cases;
  aclrtStream stream = nullptr;
  aclmdlRI graph = nullptr;
  aclScalar *one = nullptr;
  void *workspace = nullptr;
  uint64_t workspace_bytes = 0;
  size_t arena_bytes = 0;
  double max_error = 0;
  size_t checked = 0;
  explicit Runner(const char *path) : in(path) {
    if (!in)
      throw std::runtime_error("fixture open");
  }
  template <class T> T get() {
    T v{};
    if (!(in >> v))
      throw std::runtime_error("truncated fixture");
    return v;
  }
  size_t count(size_t max) {
    auto n = get<size_t>();
    if (n > max)
      throw std::runtime_error("fixture bound");
    return n;
  }
  void *alloc(size_t bytes) {
    void *p = nullptr;
    check(
        aclrtMalloc(&p, std::max(size_t(64), bytes), ACL_MEM_MALLOC_HUGE_FIRST),
        "allocate");
    allocations.push_back(p);
    return p;
  }
  std::vector<Binding> bindings() {
    std::vector<Binding> v(count(2048));
    for (auto &b : v) {
      b.id = count(buffers.size() - 1);
      auto n = count(16777216);
      if (n * 4 != buffers[b.id].bytes)
        throw std::runtime_error("binding shape");
      b.values.resize(n);
      for (auto &x : b.values) {
        x = get<float>();
        if (!std::isfinite(x))
          throw std::runtime_error("nonfinite fixture");
      }
    }
    return v;
  }
  void prepare() {
    if (get<std::string>() != "IF_NPU_QUALIFY_1")
      throw std::runtime_error("fixture version");
    arena_bytes = count(67108864);
    void *arena = alloc(arena_bytes);
    buffers.resize(count(2048));
    if (buffers.empty())
      throw std::runtime_error("no buffers");
    for (auto &b : buffers) {
      b.kind = get<int>();
      b.offset = get<int64_t>();
      b.shape.resize(count(8));
      if (b.shape.empty())
        throw std::runtime_error("rank");
      for (auto &d : b.shape) {
        d = get<int64_t>();
        if (d <= 0 || uint64_t(d) > 67108864 / b.bytes)
          throw std::runtime_error("shape bound");
        b.bytes *= d;
      }
      auto n = count(16777216);
      if (n && n * 4 != b.bytes)
        throw std::runtime_error("initial shape");
      b.initial.resize(n);
      for (auto &x : b.initial)
        x = get<float>();
      if (b.kind == 3) {
        if (b.offset < 0 || size_t(b.offset) > arena_bytes ||
            b.bytes > arena_bytes - size_t(b.offset) || b.offset % 64)
          throw std::runtime_error("arena offset");
        b.ptr = static_cast<char *>(arena) + b.offset;
      } else {
        if (b.kind < 0 || b.kind > 2)
          throw std::runtime_error("storage");
        b.ptr = alloc(b.bytes);
      }
      std::vector<int64_t> stride(b.shape.size(), 1);
      for (size_t i = stride.size() - 1; i > 0; --i)
        stride[i - 1] = stride[i] * b.shape[i];
      b.tensor = aclCreateTensor(b.shape.data(), b.shape.size(), ACL_FLOAT,
                                 stride.data(), 0, ACL_FORMAT_ND,
                                 b.shape.data(), b.shape.size(), b.ptr);
      if (!b.tensor)
        throw std::runtime_error("tensor");
    }
    float alpha = 1;
    one = aclCreateScalar(&alpha, ACL_FLOAT);
    if (!one)
      throw std::runtime_error("scalar");
    steps.resize(count(512));
    for (auto &s : steps) {
      s.op = get<int>();
      s.a = count(buffers.size() - 1);
      s.b = count(buffers.size() - 1);
      s.y = count(buffers.size() - 1);
      auto *a = buffers[s.a].tensor;
      auto *b = buffers[s.b].tensor;
      auto *y = buffers[s.y].tensor;
      switch (s.op) {
      case 1:
        s.launch = aclnnAdd;
        check(aclnnAddGetWorkspaceSize(a, b, one, y, &s.bytes, &s.executor),
              "add prepare");
        break;
      case 2:
        s.launch = aclnnMul;
        check(aclnnMulGetWorkspaceSize(a, b, y, &s.bytes, &s.executor),
              "mul prepare");
        break;
      case 3:
        s.launch = aclnnRelu;
        check(aclnnReluGetWorkspaceSize(a, y, &s.bytes, &s.executor),
              "relu prepare");
        break;
      case 4:
        s.launch = aclnnMatmul;
        check(aclnnMatmulGetWorkspaceSize(a, b, y, 0, &s.bytes, &s.executor),
              "matmul prepare");
        break;
      default:
        throw std::runtime_error("op");
      }
      check(aclSetAclOpExecutorRepeatable(s.executor), "repeatable");
      workspace_bytes = std::max(workspace_bytes, s.bytes);
    }
    workspace = alloc(workspace_bytes);
    updates.resize(count(2048));
    for (auto &u : updates) {
      u.first = count(buffers.size() - 1);
      u.second = count(buffers.size() - 1);
      if (buffers[u.first].kind != 2 ||
          buffers[u.first].bytes != buffers[u.second].bytes)
        throw std::runtime_error("state update");
      snapshots.push_back(alloc(buffers[u.first].bytes));
    }
    cases.resize(count(1024));
    if (cases.empty())
      throw std::runtime_error("no cases");
    for (auto &c : cases) {
      c.inputs = bindings();
      c.expected = bindings();
      for (auto &b : c.inputs)
        if (buffers[b.id].kind != 0)
          throw std::runtime_error("input storage");
    }
    std::string trailing;
    if (in >> trailing)
      throw std::runtime_error("trailing fixture");
  }
  void reset() {
    for (auto &b : buffers) {
      check(aclrtMemset(b.ptr, b.bytes, 0, b.bytes), "zero");
      if (!b.initial.empty())
        check(aclrtMemcpy(b.ptr, b.bytes, b.initial.data(), b.bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE),
              "initialize");
    }
  }
  void upload(const Case &c) {
    for (auto &b : c.inputs)
      check(aclrtMemcpy(buffers[b.id].ptr, buffers[b.id].bytes, b.values.data(),
                        b.values.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE),
            "input");
  }
  void launch() {
    for (auto &s : steps)
      check(s.launch(workspace, s.bytes, s.executor, stream), "enqueue");
    for (size_t i = 0; i < updates.size(); ++i) {
      auto &b = buffers[updates[i].second];
      check(aclrtMemcpyAsync(snapshots[i], b.bytes, b.ptr, b.bytes,
                             ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
            "state snapshot");
    }
    for (size_t i = 0; i < updates.size(); ++i) {
      auto &b = buffers[updates[i].first];
      check(aclrtMemcpyAsync(b.ptr, b.bytes, snapshots[i], b.bytes,
                             ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
            "state commit");
    }
  }
  void compare(const Case &c) {
    for (auto &b : c.expected) {
      std::vector<float> got(b.values.size());
      check(aclrtMemcpy(got.data(), got.size() * 4, buffers[b.id].ptr,
                        got.size() * 4, ACL_MEMCPY_DEVICE_TO_HOST),
            "read");
      for (size_t i = 0; i < got.size(); ++i) {
        double error = std::abs(double(got[i]) - b.values[i]);
        max_error = std::max(max_error, error);
        if (!std::isfinite(got[i]) ||
            error > 0.0002 + 0.0001 * std::abs(b.values[i]))
          throw std::runtime_error(
              "numerical mismatch buffer=" + std::to_string(b.id) +
              " index=" + std::to_string(i) + " got=" + std::to_string(got[i]) +
              " expected=" + std::to_string(b.values[i]));
        ++checked;
      }
    }
  }
  void run() {
    reset();
    for (auto &c : cases) {
      upload(c);
      launch();
      check(aclrtSynchronizeStream(stream), "eager fence");
      compare(c);
    }
    reset();
    upload(cases[0]);
    launch();
    check(aclrtSynchronizeStream(stream), "warmup");
    check(aclmdlRICaptureBegin(stream, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),
          "capture begin");
    try {
      launch();
    } catch (...) {
      aclmdlRICaptureEnd(stream, &graph);
      throw;
    }
    check(aclmdlRICaptureEnd(stream, &graph), "capture end");
    for (int pass = 0; pass < 3; ++pass) {
      reset();
      for (auto &c : cases) {
        upload(c);
        check(aclmdlRIExecuteAsync(graph, stream), "replay");
        check(aclrtSynchronizeStream(stream), "replay fence");
        compare(c);
      }
    }
  }
  ~Runner() {
    if (stream)
      aclrtSynchronizeStream(stream);
    if (graph)
      aclmdlRIDestroy(graph);
    for (auto &s : steps)
      if (s.executor)
        aclDestroyAclOpExecutor(s.executor);
    for (auto &b : buffers)
      if (b.tensor)
        aclDestroyTensor(b.tensor);
    if (one)
      aclDestroyScalar(one);
    for (auto p : allocations)
      aclrtFree(p);
    if (stream)
      aclrtDestroyStream(stream);
  }
};
int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: physical_plan_probe FIXTURE DEVICE\n";
    return 2;
  }
  try {
    int device = std::stoi(argv[2]);
    check(aclInit(nullptr), "init");
    check(aclrtSetDevice(device), "device");
    {
      Runner r(argv[1]);
      check(aclrtCreateStream(&r.stream), "stream");
      r.prepare();
      r.run();
      std::cout << "{\"status\":\"passed\",\"device\":" << device
                << ",\"steps\":" << r.steps.size()
                << ",\"arena_bytes\":" << r.arena_bytes
                << ",\"workspace_bytes\":" << r.workspace_bytes
                << ",\"cases\":" << r.cases.size()
                << ",\"graph_passes\":3,\"checked_values\":" << r.checked
                << ",\"max_abs_error\":" << r.max_error << "}\n";
    }
    check(aclrtResetDevice(device), "reset");
    check(aclFinalize(), "finalize");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
