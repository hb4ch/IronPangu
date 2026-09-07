#include "pangu_acl.h"
#include <acl/acl.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_copy.h>
#include <aclnnop/aclnn_cos.h>
#include <aclnnop/aclnn_embedding.h>
#include <aclnnop/aclnn_exp.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_reduce_sum.h>
#include <aclnnop/aclnn_rms_norm.h>
#include <aclnnop/aclnn_rsqrt.h>
#include <aclnnop/aclnn_s_where.h>
#include <aclnnop/aclnn_scatter_nd_update.h>
#include <aclnnop/aclnn_sigmoid.h>
#include <aclnnop/aclnn_silu.h>
#include <aclnnop/aclnn_sin.h>
#include <aclnnop/aclnn_softmax.h>
#include <aclnnop/aclnn_softplus.h>
#include <aclnnop/aclnn_sub.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <hccl/hccl.h>
#include <hccl/hccl_comm.h>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
thread_local std::string last_error;
std::mutex init_mutex;
unsigned init_users = 0;
void check(aclError status, const char *operation) {
  if (status != ACL_SUCCESS)
    throw std::runtime_error(std::string(operation) + ": " +
                             std::to_string(status));
}
template <class F> int32_t boundary(F f) noexcept {
  try {
    f();
    last_error.clear();
    return 0;
  } catch (const std::exception &e) {
    last_error = e.what();
    return -1;
  } catch (...) {
    last_error = "unknown C++ exception";
    return -1;
  }
}
} // namespace
struct pangu_acl_session {
  aclrtContext context = nullptr;
  aclrtStream stream = nullptr;
  HcclComm communicator = nullptr;
  uint32_t tp_rank = 0, tp_world = 1;
  std::thread::id thread = std::this_thread::get_id();
  struct Buffer {
    void *ptr;
    uint64_t bytes;
  };
  std::map<uint64_t, Buffer> buffers;
  std::map<uint64_t, aclmdlRI> graphs;
  uint64_t memory_budget = 0, memory_reserve = 0;
  uint64_t baseline_free = 0, minimum_free = UINT64_MAX;
  std::pair<uint64_t, uint64_t> observe_memory() {
    size_t free = 0, total = 0;
    check(aclrtGetMemInfo(ACL_HBM_MEM, &free, &total), "HBM memory info");
    minimum_free = std::min(minimum_free, uint64_t(free));
    return {free, total};
  }
  void check_allocation(uint64_t bytes) {
    if (!memory_budget)
      return;
    auto [free, total] = observe_memory();
    (void)total;
    auto used = baseline_free > free ? baseline_free - free : 0;
    if (used > memory_budget || bytes > memory_budget - used ||
        free < memory_reserve || bytes > free - memory_reserve)
      throw std::runtime_error(
          "HBM budget exceeded: observed_used=" + std::to_string(used) +
          " allocation=" + std::to_string(bytes) +
          " budget=" + std::to_string(memory_budget) +
          "; reduce --max-model-len or increase --gpu-memory-utilization");
  }
  struct Linear {
    std::vector<aclTensor *> tensors;
    std::vector<aclScalar *> scalars;
    std::vector<aclIntArray *> arrays;
    std::vector<Linear *> children;
    using Launch = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *,
                                   aclrtStream);
    struct Stage {
      std::string name;
      aclOpExecutor *executor = nullptr;
      uint64_t bytes = 0;
      Launch launch = nullptr;
    };
    std::vector<Stage> stages;
    struct DeviceCopy {
      void *dst;
      const void *src;
      uint64_t bytes;
    };
    std::vector<DeviceCopy> copies;
    struct Gather {
      void *send;
      void *receive;
      uint64_t count;
      HcclComm comm;
    };
    std::vector<Gather> gathers;
    std::vector<void *> temporaries;
    uint64_t temporary_bytes = 0;
    pangu_acl_session *owner = nullptr;
    void *workspace = nullptr;
    uint64_t workspace_bytes = 0;
    aclnnStatus run(aclrtStream stream) {
      for (auto *child : children) {
        auto status = child->run(stream);
        if (status != ACL_SUCCESS)
          return status;
      }
      for (auto &stage : stages) {
        auto status =
            stage.launch(workspace, stage.bytes, stage.executor, stream);
        if (status != ACL_SUCCESS)
          throw std::runtime_error(stage.name + ": " + std::to_string(status));
      }
      for (const auto &gather : gathers) {
        auto result = HcclAllGather(gather.send, gather.receive, gather.count,
                                    HCCL_DATA_TYPE_BFP16, gather.comm, stream);
        if (result != HCCL_SUCCESS)
          throw std::runtime_error("TP allgather: " + std::to_string(result));
      }
      for (const auto &copy : copies)
        check(aclrtMemcpyAsync(copy.dst, copy.bytes, copy.src, copy.bytes,
                               ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
              "captured state copy");
      return ACL_SUCCESS;
    }
    void prepare_workspace() {
      for (auto &stage : stages) {
        if (!stage.executor)
          throw std::runtime_error("null executor in stage " +
                                   std::to_string(&stage - stages.data()));
        workspace_bytes = std::max(workspace_bytes, stage.bytes);
      }
      if (owner)
        owner->check_allocation(workspace_bytes);
      if (workspace_bytes)
        check(
            aclrtMalloc(&workspace, workspace_bytes, ACL_MEM_MALLOC_HUGE_FIRST),
            "workspace allocate");
      if (owner && owner->memory_budget)
        owner->observe_memory();
    }
    ~Linear() {
      for (auto &stage : stages)
        if (stage.executor)
          aclDestroyAclOpExecutor(stage.executor);
      for (auto *t : tensors)
        aclDestroyTensor(t);
      for (auto *scalar : scalars)
        aclDestroyScalar(scalar);
      for (auto *array : arrays)
        aclDestroyIntArray(array);
      if (workspace)
        aclrtFree(workspace);
      for (auto *ptr : temporaries)
        aclrtFree(ptr);
    }
  };
  std::map<uint64_t, std::unique_ptr<Linear>> operations;
  uint64_t next = 1;
  bool poisoned = false;
  void enter() {
    if (thread != std::this_thread::get_id())
      throw std::runtime_error("wrong device thread");
    if (poisoned)
      throw std::runtime_error("session poisoned after capture failure");
    check(aclrtSetCurrentContext(context), "aclrtSetCurrentContext");
  }
  void *region(uint64_t id, uint64_t offset, uint64_t bytes) {
    auto it = buffers.find(id);
    if (it == buffers.end() || offset > it->second.bytes ||
        bytes > it->second.bytes - offset)
      throw std::runtime_error("invalid allocation region");
    return static_cast<char *>(it->second.ptr) + offset;
  }
};
// Builder for fixed-shape composite ACLNN operations. All allocations and
// queries happen before execution/capture; intermediate buffers and executors
// are retained.
struct Composite {
  pangu_acl_session *session;
  std::unique_ptr<pangu_acl_session::Linear> op =
      std::make_unique<pangu_acl_session::Linear>();
  explicit Composite(pangu_acl_session *s) : session(s) {
    s->enter();
    op->owner = s;
  }
  void *temp(uint64_t bytes) {
    session->check_allocation(bytes);
    void *ptr = nullptr;
    check(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST),
          "composite allocation");
    try {
      op->temporaries.push_back(ptr);
      op->temporary_bytes += bytes;
    } catch (...) {
      aclrtFree(ptr);
      throw;
    }
    return ptr;
  }
  aclTensor *tensor(void *ptr, std::vector<int64_t> shape, aclDataType dtype,
                    std::vector<int64_t> strides = {}, int64_t offset = 0,
                    std::vector<int64_t> storage = {}) {
    if (strides.empty()) {
      strides.resize(shape.size(), 1);
      for (size_t i = shape.size() - 1; i > 0; --i)
        strides[i - 1] = strides[i] * shape[i];
    }
    if (storage.empty())
      storage = shape;
    auto *t = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(),
                              offset, ACL_FORMAT_ND, storage.data(),
                              storage.size(), ptr);
    if (!t)
      throw std::runtime_error("composite tensor");
    try {
      op->tensors.push_back(t);
    } catch (...) {
      aclDestroyTensor(t);
      throw;
    }
    return t;
  }
  aclTensor *scratch(std::vector<int64_t> shape,
                     aclDataType dtype = ACL_FLOAT) {
    uint64_t count = 1;
    for (auto d : shape)
      count *= d;
    return tensor(temp(count * (dtype == ACL_FLOAT ? 4 : 2)), shape, dtype);
  }
  aclScalar *scalar(float value) {
    auto *ptr = aclCreateScalar(&value, ACL_FLOAT);
    if (!ptr)
      throw std::runtime_error("composite scalar");
    try {
      op->scalars.push_back(ptr);
    } catch (...) {
      aclDestroyScalar(ptr);
      throw;
    }
    return ptr;
  }
  template <class F>
  void step(pangu_acl_session::Linear::Launch launch, F query,
            const char *name) {
    op->stages.emplace_back();
    auto &stage = op->stages.back();
    stage.launch = launch;
    stage.name = name;
    check(query(&stage.bytes, &stage.executor), name);
    check(aclSetAclOpExecutorRepeatable(stage.executor),
          (std::string("retain ") + name).c_str());
  }
  aclTensor *cast(aclTensor *x, std::vector<int64_t> shape, aclDataType dtype) {
    auto *y = scratch(shape, dtype);
    step(
        aclnnCast,
        [&](auto *n, auto *e) {
          return aclnnCastGetWorkspaceSize(x, dtype, y, n, e);
        },
        "composite cast");
    return y;
  }
  void mul(aclTensor *a, aclTensor *b, aclTensor *y) {
    step(
        aclnnMul,
        [&](auto *n, auto *e) {
          return aclnnMulGetWorkspaceSize(a, b, y, n, e);
        },
        "composite multiply");
  }
  void matmul(aclTensor *a, aclTensor *b, aclTensor *y) {
    step(
        aclnnMatmul,
        [&](auto *n, auto *e) {
          return aclnnMatmulGetWorkspaceSize(a, b, y, 0, n, e);
        },
        "composite matmul");
  }
  void add(aclTensor *a, aclTensor *b, aclTensor *y, bool subtract = false) {
    auto *one = scalar(1.0f);
    if (subtract)
      step(
          aclnnSub,
          [&](auto *n, auto *e) {
            return aclnnSubGetWorkspaceSize(a, b, one, y, n, e);
          },
          "composite subtract");
    else
      step(
          aclnnAdd,
          [&](auto *n, auto *e) {
            return aclnnAddGetWorkspaceSize(a, b, one, y, n, e);
          },
          "composite add");
  }
  uint64_t finish() {
    op->prepare_workspace();
    auto id = session->next++;
    session->operations.emplace(id, std::move(op));
    return id;
  }
};
extern "C" {
uint32_t pangu_acl_abi_version(void) { return 2; }
const char *pangu_acl_last_error(void) { return last_error.c_str(); }
int32_t pangu_acl_open(int32_t device, pangu_acl_session **out) {
  return boundary([&] {
    if (!out)
      throw std::runtime_error("null session output");
    *out = nullptr;
    std::lock_guard<std::mutex> guard(init_mutex);
    if (init_users == 0)
      check(aclInit(nullptr), "aclInit");
    auto *s = new pangu_acl_session;
    try {
      check(aclrtCreateContext(&s->context, device), "aclrtCreateContext");
      check(aclrtCreateStream(&s->stream), "aclrtCreateStream");
      ++init_users;
      *out = s;
    } catch (...) {
      if (s->context)
        aclrtDestroyContext(s->context);
      delete s;
      if (init_users == 0)
        aclFinalize();
      throw;
    }
  });
}
int32_t pangu_acl_set_memory_budget(pangu_acl_session *s, uint64_t bytes,
                                    uint64_t reserve) {
  return boundary([&] {
    if (!s || !bytes)
      throw std::runtime_error("invalid memory budget");
    s->enter();
    auto [free, total] = s->observe_memory();
    if (bytes > total || free <= reserve)
      throw std::runtime_error("insufficient free HBM for memory reserve");
    s->memory_budget = std::min(bytes, free - reserve);
    s->memory_reserve = reserve;
    s->baseline_free = free;
    s->minimum_free = free;
  });
}
int32_t pangu_acl_memory_snapshot(pangu_acl_session *s,
                                  pangu_acl_memory_stats *out) {
  return boundary([&] {
    if (!s || !out)
      throw std::runtime_error("invalid memory snapshot");
    s->enter();
    check(aclrtSynchronizeStream(s->stream), "memory snapshot fence");
    auto [free, total] = s->observe_memory();
    *out = {};
    out->free_bytes = free;
    out->total_bytes = total;
    out->observed_peak_bytes = s->baseline_free > s->minimum_free
                                   ? s->baseline_free - s->minimum_free
                                   : 0;
    out->budget_bytes = s->memory_budget;
    out->graphs = s->graphs.size();
    for (auto &[id, b] : s->buffers) {
      (void)id;
      out->buffer_bytes += b.bytes;
    }
    for (auto &[id, op] : s->operations) {
      (void)id;
      out->temporary_bytes += op->temporary_bytes;
      out->workspace_bytes += op->workspace_bytes;
    }
    s->check_allocation(0);
  });
}
int32_t pangu_acl_allocate(pangu_acl_session *s, uint64_t bytes,
                           uint64_t *handle) {
  return boundary([&] {
    if (!s || !handle || bytes == 0 ||
        bytes > std::numeric_limits<size_t>::max())
      throw std::runtime_error("invalid allocation");
    s->enter();
    s->check_allocation(bytes);
    void *ptr = nullptr;
    check(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    try {
      auto id = s->next++;
      s->buffers.emplace(id, pangu_acl_session::Buffer{ptr, bytes});
      *handle = id;
      if (s->memory_budget)
        s->observe_memory();
    } catch (...) {
      aclrtFree(ptr);
      throw;
    }
  });
}
int32_t pangu_acl_write(pangu_acl_session *s, uint64_t id, uint64_t offset,
                        const void *host, uint64_t bytes) {
  return boundary([&] {
    if (!s || !host)
      throw std::runtime_error("null write argument");
    s->enter();
    check(aclrtMemcpy(s->region(id, offset, bytes), bytes, host, bytes,
                      ACL_MEMCPY_HOST_TO_DEVICE),
          "aclrtMemcpy H2D");
  });
}
int32_t pangu_acl_read(pangu_acl_session *s, uint64_t id, uint64_t offset,
                       void *host, uint64_t bytes) {
  return boundary([&] {
    if (!s || !host)
      throw std::runtime_error("null read argument");
    s->enter();
    check(aclrtMemcpy(host, bytes, s->region(id, offset, bytes), bytes,
                      ACL_MEMCPY_DEVICE_TO_HOST),
          "aclrtMemcpy D2H");
  });
}
int32_t pangu_acl_capture_copy(pangu_acl_session *s, uint64_t source,
                               uint64_t destination, uint64_t bytes,
                               uint64_t *graph) {
  return boundary([&] {
    if (!s || !graph || source == destination || bytes == 0)
      throw std::runtime_error("invalid graph arguments");
    s->enter();
    void *src = s->region(source, 0, bytes),
         *dst = s->region(destination, 0, bytes);
    check(aclrtMemcpyAsync(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE,
                           s->stream),
          "warmup copy");
    check(aclrtSynchronizeStream(s->stream), "warmup fence");
    check(aclmdlRICaptureBegin(s->stream, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),
          "capture begin");
    auto status = aclrtMemcpyAsync(dst, bytes, src, bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, s->stream);
    aclmdlRI model = nullptr;
    auto end = aclmdlRICaptureEnd(s->stream, &model);
    if (status != ACL_SUCCESS || end != ACL_SUCCESS) {
      s->poisoned = true;
      if (model)
        aclmdlRIDestroy(model);
      check(status, "captured copy");
      check(end, "capture end");
    }
    try {
      auto id = s->next++;
      s->graphs.emplace(id, model);
      *graph = id;
    } catch (...) {
      aclmdlRIDestroy(model);
      throw;
    }
  });
}
int32_t pangu_acl_replay(pangu_acl_session *s, uint64_t graph) {
  return boundary([&] {
    if (!s)
      throw std::runtime_error("null session");
    s->enter();
    auto it = s->graphs.find(graph);
    if (it == s->graphs.end())
      throw std::runtime_error("unknown graph");
    check(aclmdlRIExecuteAsync(it->second, s->stream), "graph replay");
    check(aclrtSynchronizeStream(s->stream), "replay fence");
    s->check_allocation(0);
  });
}

int32_t pangu_acl_device_count(uint32_t *count) {
  return boundary([&] {
    if (!count)
      throw std::runtime_error("null device count");
    check(aclrtGetDeviceCount(count), "TP device count");
  });
}
uint64_t pangu_acl_tp_root_size(void) { return sizeof(HcclRootInfo); }
int32_t pangu_acl_tp_root(void *bytes, uint64_t size) {
  return boundary([&] {
    if (!bytes || size != sizeof(HcclRootInfo))
      throw std::runtime_error("invalid TP root buffer");
    const char *mode = std::getenv("HCCL_OP_EXPANSION_MODE");
    if (!mode || std::string(mode) != "HOST")
      throw std::runtime_error(
          "TP ACL graph requires HCCL_OP_EXPANSION_MODE=HOST on this CANN "
          "image");
    auto result = HcclGetRootInfo(static_cast<HcclRootInfo *>(bytes));
    if (result != HCCL_SUCCESS)
      throw std::runtime_error("TP root: " + std::to_string(result));
  });
}
int32_t pangu_acl_tp_init(pangu_acl_session *s, uint32_t world, uint32_t rank,
                          const void *root, uint64_t size) {
  return boundary([&] {
    if (!s || !root || size != sizeof(HcclRootInfo) || world < 2 || world > 8 ||
        rank >= world || s->communicator)
      throw std::runtime_error("invalid TP communicator");
    s->enter();
    // Qualified HOST expansion is selected before HCCL root creation.
    auto result = HcclCommInitRootInfo(
        world, static_cast<const HcclRootInfo *>(root), rank, &s->communicator);
    if (result != HCCL_SUCCESS)
      throw std::runtime_error("TP init: " + std::to_string(result));
    s->tp_world = world;
    s->tp_rank = rank;
  });
}
int32_t pangu_acl_linear_prepare(pangu_acl_session *s, uint64_t x, uint64_t w,
                                 uint64_t y, int64_t m, int64_t n, int64_t k,
                                 uint64_t *operation) {
  return boundary([&] {
    if (!s || !operation || m <= 0 || n <= 0 || k <= 0 || m > 1048576 ||
        n > 1048576 || k > 1048576 || x == y || w == y)
      throw std::runtime_error("invalid linear arguments");
    s->enter();
    auto op = std::make_unique<pangu_acl_session::Linear>();
    op->owner = s;
    auto tensor = [&](void *ptr, int64_t rows, int64_t cols, bool transpose) {
      int64_t storage[] = {rows, cols};
      int64_t shape[] = {transpose ? cols : rows, transpose ? rows : cols};
      int64_t strides[] = {transpose ? 1 : cols, transpose ? cols : 1};
      aclTensor *t = aclCreateTensor(shape, 2, ACL_BF16, strides, 0,
                                     ACL_FORMAT_ND, storage, 2, ptr);
      if (!t)
        throw std::runtime_error("aclCreateTensor");
      try {
        op->tensors.push_back(t);
      } catch (...) {
        aclDestroyTensor(t);
        throw;
      }
      return t;
    };
    auto *a = tensor(s->region(x, 0, uint64_t(m) * k * 2), m, k, false);
    if (n % s->tp_world)
      throw std::runtime_error("matmul output width is not divisible by TP");
    int64_t local_n = n / s->tp_world;
    // Embedding remains replicated for lookup and is sliced only for the tied
    // LM head.
    uint64_t weight_offset = s->buffers.at(w).bytes == uint64_t(n) * k * 2
                                 ? uint64_t(s->tp_rank) * local_n * k * 2
                                 : 0;
    auto *b = tensor(s->region(w, weight_offset, uint64_t(local_n) * k * 2),
                     local_n, k, true);
    void *result = s->region(y, 0, uint64_t(m) * n * 2);
    void *local_result = result;
    if (s->tp_world > 1) {
      auto temporary = [&](uint64_t bytes) {
        s->check_allocation(bytes);
        void *ptr = nullptr;
        check(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST),
              "TP temporary");
        op->temporaries.push_back(ptr);
        op->temporary_bytes += bytes;
        return ptr;
      };
      local_result = temporary(uint64_t(m) * local_n * 2);
      auto *gathered = static_cast<char *>(temporary(uint64_t(m) * n * 2));
      op->gathers.push_back(
          {local_result, gathered, uint64_t(m) * local_n, s->communicator});
      // HCCL returns [rank, batch, local_n]; restore canonical [batch, n].
      for (uint32_t rank = 0; rank < s->tp_world; ++rank)
        for (int64_t row = 0; row < m; ++row)
          op->copies.push_back(
              {static_cast<char *>(result) + (row * n + rank * local_n) * 2,
               gathered + (rank * m + row) * local_n * 2,
               uint64_t(local_n) * 2});
    }
    auto *out = tensor(local_result, m, local_n, false);
    op->stages.resize(1);
    auto &stage = op->stages[0];
    stage.launch = aclnnMatmul;
    check(aclnnMatmulGetWorkspaceSize(a, b, out, 0, &stage.bytes,
                                      &stage.executor),
          "linear workspace");
    check(aclSetAclOpExecutorRepeatable(stage.executor),
          "retain linear executor");
    op->prepare_workspace();
    auto id = s->next++;
    s->operations.emplace(id, std::move(op));
    *operation = id;
  });
}
int32_t pangu_acl_operation_execute(pangu_acl_session *s, uint64_t operation) {
  return boundary([&] {
    if (!s)
      throw std::runtime_error("null session");
    s->enter();
    auto it = s->operations.find(operation);
    if (it == s->operations.end())
      throw std::runtime_error("unknown linear operation");
    auto &op = *it->second;
    check(op.run(s->stream), "linear execute");
    check(aclrtSynchronizeStream(s->stream), "linear fence");
  });
}
int32_t pangu_acl_operation_capture(pangu_acl_session *s, uint64_t operation,
                                    uint64_t *graph) {
  return boundary([&] {
    if (!s || !graph)
      throw std::runtime_error("null capture argument");
    s->enter();
    auto it = s->operations.find(operation);
    if (it == s->operations.end())
      throw std::runtime_error("unknown linear operation");
    auto &op = *it->second;
    check(op.run(s->stream), "linear warmup");
    check(aclrtSynchronizeStream(s->stream), "linear warmup fence");
    check(aclmdlRICaptureBegin(s->stream, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),
          "linear capture begin");
    aclnnStatus status = ACL_SUCCESS;
    std::exception_ptr capture_error;
    try {
      status = op.run(s->stream);
    } catch (...) {
      capture_error = std::current_exception();
    }
    // Close capture even when an ACLNN/HCCL launch throws. Never reuse a failed
    // session.
    aclmdlRI model = nullptr;
    auto end = aclmdlRICaptureEnd(s->stream, &model);
    if (capture_error || status != ACL_SUCCESS || end != ACL_SUCCESS) {
      s->poisoned = true;
      if (model)
        aclmdlRIDestroy(model);
      if (capture_error)
        std::rethrow_exception(capture_error);
      check(status, "captured linear");
      check(end, "linear capture end");
    }
    try {
      auto id = s->next++;
      s->graphs.emplace(id, model);
      *graph = id;
    } catch (...) {
      aclmdlRIDestroy(model);
      throw;
    }
  });
}

int32_t pangu_acl_linear_execute(pangu_acl_session *s, uint64_t id) {
  return pangu_acl_operation_execute(s, id);
}
int32_t pangu_acl_linear_capture(pangu_acl_session *s, uint64_t id,
                                 uint64_t *graph) {
  return pangu_acl_operation_capture(s, id, graph);
}
int32_t pangu_acl_rms_prepare(pangu_acl_session *s, uint64_t x, uint64_t gamma,
                              uint64_t y, int64_t m, int64_t k,
                              uint64_t *operation) {
  return boundary([&] {
    if (!s || !operation || m <= 0 || k <= 0 || m > 1048576 || k > 1048576 ||
        x == y || gamma == y)
      throw std::runtime_error("invalid RMS arguments");
    s->enter();
    auto op = std::make_unique<pangu_acl_session::Linear>();
    op->owner = s;
    auto temp = [&](uint64_t bytes) {
      s->check_allocation(bytes);
      void *ptr = nullptr;
      check(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST),
            "RMS temporary");
      try {
        op->temporaries.push_back(ptr);
        op->temporary_bytes += bytes;
      } catch (...) {
        aclrtFree(ptr);
        throw;
      }
      return ptr;
    };
    auto tensor = [&](void *ptr, std::vector<int64_t> shape,
                      aclDataType dtype) {
      std::vector<int64_t> strides(shape.size(), 1);
      for (size_t i = shape.size() - 1; i > 0; --i)
        strides[i - 1] = strides[i] * shape[i];
      auto *t =
          aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                          ACL_FORMAT_ND, shape.data(), shape.size(), ptr);
      if (!t)
        throw std::runtime_error("RMS tensor descriptor");
      try {
        op->tensors.push_back(t);
      } catch (...) {
        aclDestroyTensor(t);
        throw;
      }
      return t;
    };
    auto *input =
        tensor(s->region(x, 0, uint64_t(m) * k * 2), {m, k}, ACL_BF16);
    auto *scale = tensor(s->region(gamma, 0, uint64_t(k) * 4), {k}, ACL_FLOAT);
    auto *output =
        tensor(s->region(y, 0, uint64_t(m) * k * 2), {m, k}, ACL_BF16);
    auto *xf = tensor(temp(uint64_t(m) * k * 4), {m, k}, ACL_FLOAT);
    auto *yf = tensor(temp(uint64_t(m) * k * 4), {m, k}, ACL_FLOAT);
    auto *rstd = tensor(temp(uint64_t(m) * 4), {m, 1}, ACL_FLOAT);
    op->stages.resize(3);
    auto &a = op->stages[0];
    a.launch = aclnnCast;
    check(
        aclnnCastGetWorkspaceSize(input, ACL_FLOAT, xf, &a.bytes, &a.executor),
        "RMS input cast");
    check(aclSetAclOpExecutorRepeatable(a.executor),
          "retain RMS input cast executor");
    auto &b = op->stages[1];
    b.launch = aclnnRmsNorm;
    check(aclnnRmsNormGetWorkspaceSize(xf, scale, 1e-6, yf, rstd, &b.bytes,
                                       &b.executor),
          "FP32 RMS workspace");
    check(aclSetAclOpExecutorRepeatable(b.executor),
          "retain FP32 RMS workspace executor");
    auto &c = op->stages[2];
    c.launch = aclnnCast;
    check(
        aclnnCastGetWorkspaceSize(yf, ACL_BF16, output, &c.bytes, &c.executor),
        "RMS output cast");
    check(aclSetAclOpExecutorRepeatable(c.executor),
          "retain RMS output cast executor");
    op->prepare_workspace();
    auto id = s->next++;
    s->operations.emplace(id, std::move(op));
    *operation = id;
  });
}

int32_t pangu_acl_delta_prepare(pangu_acl_session *s, uint64_t q, uint64_t k,
                                uint64_t v, uint64_t g, uint64_t beta,
                                uint64_t state, uint64_t out, int64_t bh,
                                int64_t dk, int64_t dv, uint64_t *operation) {
  return boundary([&] {
    if (!s || !operation || bh <= 0 || bh > 256 || dk <= 0 || dk > 512 ||
        dv <= 0 || dv > 512)
      throw std::runtime_error("invalid delta dimensions");
    for (auto id : {q, k, v, g, beta})
      if (id == state || id == out)
        throw std::runtime_error("delta state/output alias");
    if (state == out)
      throw std::runtime_error("delta output aliases state");
    Composite b(s);
    auto *qt = b.tensor(s->region(q, 0, bh * dk * 2), {bh, 1, dk}, ACL_BF16);
    auto *kt = b.tensor(s->region(k, 0, bh * dk * 2), {bh, 1, dk}, ACL_BF16);
    auto *vt = b.tensor(s->region(v, 0, bh * dv * 2), {bh, 1, dv}, ACL_BF16);
    auto *gt = b.tensor(s->region(g, 0, bh * 4), {bh, 1, 1}, ACL_FLOAT);
    auto *bt = b.tensor(s->region(beta, 0, bh * 2), {bh, 1, 1}, ACL_BF16);
    auto *st = b.tensor(s->region(state, 0, bh * dk * dv * 4), {bh, dk, dv},
                        ACL_FLOAT);
    auto *ot = b.tensor(s->region(out, 0, bh * dv * 2), {bh, 1, dv}, ACL_BF16);
    auto *qf = b.cast(qt, {bh, 1, dk}, ACL_FLOAT);
    // Two views of one FP32 key allocation, so the outer product never
    // transposes state.
    void *key_ptr = b.temp(bh * dk * 4);
    auto *kf = b.tensor(key_ptr, {bh, 1, dk}, ACL_FLOAT);
    auto *kc = b.tensor(key_ptr, {bh, dk, 1}, ACL_FLOAT);
    b.step(
        aclnnCast,
        [&](auto *n, auto *e) {
          return aclnnCastGetWorkspaceSize(kt, ACL_FLOAT, kf, n, e);
        },
        "delta key cast");
    auto *vf = b.cast(vt, {bh, 1, dv}, ACL_FLOAT);
    auto *bf = b.cast(bt, {bh, 1, 1}, ACL_FLOAT);
    auto *decay = b.scratch({bh, 1, 1});
    b.step(
        aclnnExp,
        [&](auto *n, auto *e) {
          return aclnnExpGetWorkspaceSize(gt, decay, n, e);
        },
        "delta exp unclamped");
    auto *decayed = b.scratch({bh, dk, dv});
    b.mul(st, decay, decayed);
    auto *pred = b.scratch({bh, 1, dv});
    b.matmul(kf, decayed, pred);
    auto *residual = b.scratch({bh, 1, dv});
    b.add(vf, pred, residual, true);
    auto *weighted = b.scratch({bh, 1, dv});
    b.mul(residual, bf, weighted);
    auto *outer = b.scratch({bh, dk, dv});
    b.matmul(kc, weighted, outer);
    b.add(decayed, outer, st);
    float scale = 1.0f / std::sqrt(static_cast<float>(dk));
    void *scale_ptr = b.temp(4);
    check(aclrtMemcpy(scale_ptr, 4, &scale, 4, ACL_MEMCPY_HOST_TO_DEVICE),
          "delta scale upload");
    auto *scale_t = b.tensor(scale_ptr, {1}, ACL_FLOAT);
    auto *qs = b.scratch({bh, 1, dk});
    b.mul(qf, scale_t, qs);
    auto *result = b.scratch({bh, 1, dv});
    b.matmul(qs, st, result);
    b.step(
        aclnnCast,
        [&](auto *n, auto *e) {
          return aclnnCastGetWorkspaceSize(result, ACL_BF16, ot, n, e);
        },
        "delta output cast");
    *operation = b.finish();
  });
}

int32_t pangu_acl_conv_prepare(pangu_acl_session *s, uint64_t x,
                               uint64_t weight, uint64_t history, uint64_t out,
                               int64_t batch, int64_t channels,
                               uint64_t *operation) {
  return boundary([&] {
    if (!s || !operation || batch <= 0 || batch > 64 || channels <= 0 ||
        channels > 65536 || x == history || weight == history ||
        out == history || out == x || out == weight)
      throw std::runtime_error("invalid convolution arguments");
    Composite b(s);
    auto *xt = b.tensor(s->region(x, 0, batch * channels * 2),
                        {batch, channels}, ACL_BF16);
    auto *ot = b.tensor(s->region(out, 0, batch * channels * 2),
                        {batch, channels}, ACL_BF16);
    auto *wp = s->region(weight, 0, channels * 4 * 2);
    auto *hp = s->region(history, 0, batch * channels * 3 * 2);
    std::vector<aclTensor *> h, hf, products;
    for (int j = 0; j < 4; ++j) {
      auto *wt = b.tensor(wp, {1, channels}, ACL_BF16, {channels * 4, 4}, j,
                          {channels, 4});
      auto *wf = b.cast(wt, {1, channels}, ACL_FLOAT);
      aclTensor *input = nullptr;
      if (j < 3) {
        auto *ht = b.tensor(hp, {batch, channels}, ACL_BF16, {channels * 3, 3},
                            j, {batch, channels, 3});
        h.push_back(ht);
        input = b.cast(ht, {batch, channels}, ACL_FLOAT);
        hf.push_back(input);
      } else
        input = b.cast(xt, {batch, channels}, ACL_FLOAT);
      auto *product = b.scratch({batch, channels});
      b.mul(input, wf, product);
      products.push_back(product);
    }
    auto *sum = products[0];
    for (int j = 1; j < 4; ++j) {
      auto *next = b.scratch({batch, channels});
      b.add(sum, products[j], next);
      sum = next;
    }
    auto *rounded = b.cast(sum, {batch, channels}, ACL_BF16);
    b.step(
        aclnnSilu,
        [&](auto *n, auto *e) {
          return aclnnSiluGetWorkspaceSize(rounded, ot, n, e);
        },
        "convolution SiLU");
    // Copy from snapshots, so writes cannot overwrite the history required by
    // another stage.
    for (int j = 0; j < 2; ++j)
      b.step(
          aclnnCast,
          [&, j](auto *n, auto *e) {
            return aclnnCastGetWorkspaceSize(hf[j + 1], ACL_BF16, h[j], n, e);
          },
          "shift convolution history");
    b.step(
        aclnnInplaceCopy,
        [&](auto *n, auto *e) {
          return aclnnInplaceCopyGetWorkspaceSize(h[2], xt, n, e);
        },
        "append convolution input");
    *operation = b.finish();
  });
}

int32_t pangu_acl_sequence_prepare(pangu_acl_session *s, const uint64_t *ids,
                                   uint64_t count, uint64_t *operation) {
  return boundary([&] {
    if (!s || !ids || !operation || count == 0 || count > 4096)
      throw std::runtime_error("invalid sequence");
    s->enter();
    auto op = std::make_unique<pangu_acl_session::Linear>();
    op->owner = s;
    for (uint64_t i = 0; i < count; ++i) {
      auto it = s->operations.find(ids[i]);
      if (it == s->operations.end())
        throw std::runtime_error("unknown child operation");
      op->children.push_back(it->second.get());
    }
    auto id = s->next++;
    s->operations.emplace(id, std::move(op));
    *operation = id;
  });
}
int32_t pangu_acl_pointwise_prepare(pangu_acl_session *s, uint64_t a,
                                    uint64_t other, uint64_t out, int64_t count,
                                    int32_t kind, uint64_t *operation) {
  return boundary([&] {
    if (!s || !operation || count <= 0 || count > 16777216 || kind < 0 ||
        kind > 2 || out == a || (kind != 2 && out == other))
      throw std::runtime_error("invalid pointwise");
    Composite b(s);
    auto *at = b.tensor(s->region(a, 0, count * 2), {count}, ACL_BF16);
    auto *ot = b.tensor(s->region(out, 0, count * 2), {count}, ACL_BF16);
    if (kind == 2)
      b.step(
          aclnnSilu,
          [&](auto *n, auto *e) {
            return aclnnSiluGetWorkspaceSize(at, ot, n, e);
          },
          "pointwise SiLU");
    else {
      auto *bt = b.tensor(s->region(other, 0, count * 2), {count}, ACL_BF16);
      if (kind == 0)
        b.add(at, bt, ot);
      else
        b.mul(at, bt, ot);
    }
    *operation = b.finish();
  });
}
int32_t pangu_acl_delta_transforms_prepare(pangu_acl_session *s, uint64_t conv,
                                           uint64_t a, uint64_t bin,
                                           uint64_t a_log, uint64_t dt_bias,
                                           uint64_t q, uint64_t k, uint64_t v,
                                           uint64_t g, uint64_t beta,
                                           int64_t batch, uint64_t *operation) {
  return boundary([&] {
    if (!s || !operation || batch <= 0 || batch > 64)
      throw std::runtime_error("invalid delta transform batch");
    const int64_t h = 16, d = 128, c = 6144;
    Composite b(s);
    auto *cp = s->region(conv, 0, batch * c * 2);
    for (int section = 0; section < 3; ++section) {
      auto *in = b.tensor(cp, {batch, h, d}, ACL_BF16, {c, d, 1},
                          section * h * d, {batch, c});
      auto *out = b.tensor(s->region(section == 0   ? q
                                     : section == 1 ? k
                                                    : v,
                                     0, batch * h * d * 2),
                           {batch, h, d}, ACL_BF16);
      if (section == 2) {
        auto *vf = b.cast(in, {batch, h, d}, ACL_FLOAT);
        b.step(
            aclnnCast,
            [&](auto *n, auto *e) {
              return aclnnCastGetWorkspaceSize(vf, ACL_BF16, out, n, e);
            },
            "delta V unpack");
        continue;
      }
      auto *square = b.scratch({batch, h, d}, ACL_BF16);
      b.mul(in, in, square);
      int64_t axis = -1;
      auto *axes = aclCreateIntArray(&axis, 1);
      if (!axes)
        throw std::runtime_error("L2 axis");
      b.op->arrays.push_back(axes);
      auto *sum = b.scratch({batch, h, 1}, ACL_BF16);
      b.step(
          aclnnReduceSum,
          [&](auto *n, auto *e) {
            return aclnnReduceSumGetWorkspaceSize(square, axes, true, ACL_BF16,
                                                  sum, n, e);
          },
          "BF16 L2 reduction");
      auto *shift = b.scratch({batch, h, 1}, ACL_BF16);
      auto *eps = b.scalar(1e-6f);
      auto *one = b.scalar(1.f);
      b.step(
          aclnnAdds,
          [&](auto *n, auto *e) {
            return aclnnAddsGetWorkspaceSize(sum, eps, one, shift, n, e);
          },
          "BF16 L2 epsilon");
      auto *inv = b.scratch({batch, h, 1}, ACL_BF16);
      b.step(
          aclnnRsqrt,
          [&](auto *n, auto *e) {
            return aclnnRsqrtGetWorkspaceSize(shift, inv, n, e);
          },
          "BF16 L2 rsqrt");
      b.mul(in, inv, out);
    }
    auto *at = b.tensor(s->region(a, 0, batch * h * 2), {batch, h}, ACL_BF16);
    auto *bt = b.tensor(s->region(bin, 0, batch * h * 2), {batch, h}, ACL_BF16);
    auto *dt = b.tensor(s->region(dt_bias, 0, h * 2), {1, h}, ACL_BF16);
    auto *al = b.tensor(s->region(a_log, 0, h * 4), {1, h}, ACL_FLOAT);
    auto *gout =
        b.tensor(s->region(g, 0, batch * h * 4), {batch, h}, ACL_FLOAT);
    auto *bout =
        b.tensor(s->region(beta, 0, batch * h * 2), {batch, h}, ACL_BF16);
    b.step(
        aclnnSigmoid,
        [&](auto *n, auto *e) {
          return aclnnSigmoidGetWorkspaceSize(bt, bout, n, e);
        },
        "delta beta");
    auto *af = b.cast(at, {batch, h}, ACL_FLOAT);
    auto *df = b.cast(dt, {1, h}, ACL_FLOAT);
    auto *sum = b.scratch({batch, h});
    b.add(af, df, sum);
    auto *soft = b.scratch({batch, h});
    auto *one = b.scalar(1.f);
    auto *threshold = b.scalar(20.f);
    b.step(
        aclnnSoftplus,
        [&](auto *n, auto *e) {
          return aclnnSoftplusGetWorkspaceSize(sum, one, threshold, soft, n, e);
        },
        "delta softplus");
    auto *ae = b.scratch({1, h});
    b.step(
        aclnnExp,
        [&](auto *n, auto *e) {
          return aclnnExpGetWorkspaceSize(al, ae, n, e);
        },
        "delta A exp");
    auto *neg = b.scratch({1, h});
    auto *minus = b.scalar(-1.f);
    b.step(
        aclnnMuls,
        [&](auto *n, auto *e) {
          return aclnnMulsGetWorkspaceSize(ae, minus, neg, n, e);
        },
        "delta negative A");
    b.mul(soft, neg, gout);
    *operation = b.finish();
  });
}
int32_t pangu_acl_gated_rms_prepare(pangu_acl_session *s, uint64_t x,
                                    uint64_t z, uint64_t weight, uint64_t out,
                                    int64_t bh, int64_t width,
                                    uint64_t *operation) {
  return boundary([&] {
    if (!s || !operation || bh <= 0 || bh > 1024 || width <= 0 || width > 2048)
      throw std::runtime_error("invalid gated norm");
    Composite b(s);
    auto *xt = b.tensor(s->region(x, 0, bh * width * 2), {bh, width}, ACL_BF16);
    auto *zt = b.tensor(s->region(z, 0, bh * width * 2), {bh, width}, ACL_BF16);
    auto *wt = b.tensor(s->region(weight, 0, width * 4), {width}, ACL_FLOAT);
    auto *ot =
        b.tensor(s->region(out, 0, bh * width * 2), {bh, width}, ACL_BF16);
    auto *xf = b.cast(xt, {bh, width}, ACL_FLOAT);
    auto *zf = b.cast(zt, {bh, width}, ACL_FLOAT);
    std::vector<float> ones(width, 1.f);
    auto *ptr = b.temp(width * 4);
    check(aclrtMemcpy(ptr, width * 4, ones.data(), width * 4,
                      ACL_MEMCPY_HOST_TO_DEVICE),
          "gated norm ones");
    auto *gamma = b.tensor(ptr, {width}, ACL_FLOAT);
    auto *norm = b.scratch({bh, width});
    auto *rstd = b.scratch({bh, 1});
    b.step(
        aclnnRmsNorm,
        [&](auto *n, auto *e) {
          return aclnnRmsNormGetWorkspaceSize(xf, gamma, 1e-6, norm, rstd, n,
                                              e);
        },
        "gated norm RMS");
    auto *rounded = b.cast(norm, {bh, width}, ACL_BF16);
    auto *normalized = b.cast(rounded, {bh, width}, ACL_FLOAT);
    auto *weighted = b.scratch({bh, width});
    b.mul(normalized, wt, weighted);
    auto *gate = b.scratch({bh, width});
    b.step(
        aclnnSilu,
        [&](auto *n, auto *e) {
          return aclnnSiluGetWorkspaceSize(zf, gate, n, e);
        },
        "gated norm SiLU");
    auto *result = b.scratch({bh, width});
    b.mul(weighted, gate, result);
    b.step(
        aclnnCast,
        [&](auto *n, auto *e) {
          return aclnnCastGetWorkspaceSize(result, ACL_BF16, ot, n, e);
        },
        "gated norm output");
    *operation = b.finish();
  });
}

int32_t pangu_acl_attention_qk_prepare(pangu_acl_session *s, uint64_t packed_q,
                                       uint64_t kin, uint64_t qgamma,
                                       uint64_t kgamma, uint64_t positions,
                                       uint64_t qout, uint64_t kout,
                                       uint64_t gate, int64_t batch,
                                       uint64_t *operation) {
  return boundary([&] {
    if (!s || !operation || batch <= 0 || batch > 64)
      throw std::runtime_error("invalid QK batch");
    Composite b(s);
    auto *pos =
        b.tensor(s->region(positions, 0, batch * 8), {batch, 1, 1}, ACL_INT64);
    auto *pf = b.cast(pos, {batch, 1, 1}, ACL_FLOAT);
    std::vector<float> freqs(32);
    for (int i = 0; i < 32; ++i)
      freqs[i] = 1.f / std::pow(10000000.f, static_cast<float>(i) / 32.f);
    void *fp = b.temp(128);
    check(aclrtMemcpy(fp, 128, freqs.data(), 128, ACL_MEMCPY_HOST_TO_DEVICE),
          "RoPE frequencies");
    auto *ft = b.tensor(fp, {1, 1, 32}, ACL_FLOAT);
    auto *angles = b.scratch({batch, 1, 32});
    b.mul(pf, ft, angles);
    auto *cosf = b.scratch({batch, 1, 32});
    auto *sinf = b.scratch({batch, 1, 32});
    b.step(
        aclnnCos,
        [&](auto *n, auto *e) {
          return aclnnCosGetWorkspaceSize(angles, cosf, n, e);
        },
        "RoPE cosine");
    b.step(
        aclnnSin,
        [&](auto *n, auto *e) {
          return aclnnSinGetWorkspaceSize(angles, sinf, n, e);
        },
        "RoPE sine");
    auto *cos = b.cast(cosf, {batch, 1, 32}, ACL_BF16);
    auto *sin = b.cast(sinf, {batch, 1, 32}, ACL_BF16);
    for (int which = 0; which < 2; ++which) {
      int64_t heads = which == 0 ? 8 : 2;
      int64_t stride = which == 0 ? 512 : 256;
      int64_t row = heads * stride;
      auto *raw = s->region(which == 0 ? packed_q : kin, 0, batch * row * 2);
      auto *in = b.tensor(raw, {batch, heads, 256}, ACL_BF16, {row, stride, 1},
                          0, {batch, row});
      auto *xf = b.cast(in, {batch, heads, 256}, ACL_FLOAT);
      auto *gamma =
          b.tensor(s->region(which == 0 ? qgamma : kgamma, 0, 256 * 4), {256},
                   ACL_FLOAT);
      auto *normf = b.scratch({batch, heads, 256});
      auto *rstd = b.scratch({batch, heads, 1});
      b.step(
          aclnnRmsNorm,
          [&](auto *n, auto *e) {
            return aclnnRmsNormGetWorkspaceSize(xf, gamma, 1e-6, normf, rstd, n,
                                                e);
          },
          "QK zero-centered RMS");
      auto *np = b.temp(batch * heads * 256 * 2);
      auto *norm = b.tensor(np, {batch, heads, 256}, ACL_BF16);
      b.step(
          aclnnCast,
          [&](auto *n, auto *e) {
            return aclnnCastGetWorkspaceSize(normf, ACL_BF16, norm, n, e);
          },
          "QK norm rounding");
      auto *outp =
          s->region(which == 0 ? qout : kout, 0, batch * heads * 256 * 2);
      auto view = [&](void *ptr, int64_t offset, int64_t width) {
        return b.tensor(ptr, {batch, heads, width}, ACL_BF16,
                        {heads * 256, 256, 1}, offset, {batch, heads, 256});
      };
      auto *real = view(np, 0, 32);
      auto *imag = view(np, 32, 32);
      auto *rc = b.scratch({batch, heads, 32}, ACL_BF16);
      auto *is = b.scratch({batch, heads, 32}, ACL_BF16);
      b.mul(real, cos, rc);
      b.mul(imag, sin, is);
      b.add(rc, is, view(outp, 0, 32), true);
      auto *ic = b.scratch({batch, heads, 32}, ACL_BF16);
      auto *rs = b.scratch({batch, heads, 32}, ACL_BF16);
      b.mul(imag, cos, ic);
      b.mul(real, sin, rs);
      b.add(ic, rs, view(outp, 32, 32));
      auto *pass = b.cast(view(np, 64, 192), {batch, heads, 192}, ACL_FLOAT);
      auto *passout = view(outp, 64, 192);
      b.step(
          aclnnCast,
          [&](auto *n, auto *e) {
            return aclnnCastGetWorkspaceSize(pass, ACL_BF16, passout, n, e);
          },
          "QK pass-through channels");
    }
    auto *gt =
        b.tensor(s->region(packed_q, 0, batch * 4096 * 2), {batch, 8, 256},
                 ACL_BF16, {4096, 512, 1}, 256, {batch, 4096});
    auto *go = b.tensor(s->region(gate, 0, batch * 2048 * 2), {batch, 8, 256},
                        ACL_BF16);
    b.step(
        aclnnSigmoid,
        [&](auto *n, auto *e) {
          return aclnnSigmoidGetWorkspaceSize(gt, go, n, e);
        },
        "attention output gate");
    *operation = b.finish();
  });
}
int32_t pangu_acl_paged_attention_prepare(
    pangu_acl_session *s, uint64_t q, uint64_t k, uint64_t v, uint64_t gate,
    uint64_t keys, uint64_t values, uint64_t append, uint64_t table,
    uint64_t mask, uint64_t out, int64_t batch, int64_t slots, int64_t context,
    uint64_t *operation) {
  return boundary([&] {
    if (!s || !operation || batch <= 0 || batch > 64 || slots <= 0 ||
        slots > 1048576 || context <= 0 || context > 262144)
      throw std::runtime_error("invalid attention capacity");
    Composite b(s);
    auto *qt = b.tensor(s->region(q, 0, batch * 2048 * 2), {batch, 2, 4, 256},
                        ACL_BF16);
    auto *kt =
        b.tensor(s->region(k, 0, batch * 512 * 2), {batch, 512}, ACL_BF16);
    auto *vt =
        b.tensor(s->region(v, 0, batch * 512 * 2), {batch, 512}, ACL_BF16);
    auto *kc =
        b.tensor(s->region(keys, 0, slots * 512 * 2), {slots, 512}, ACL_BF16);
    auto *vc =
        b.tensor(s->region(values, 0, slots * 512 * 2), {slots, 512}, ACL_BF16);
    auto *indices =
        b.tensor(s->region(append, 0, batch * 8), {batch, 1}, ACL_INT64);
    b.step(
        aclnnScatterNdUpdate,
        [&](auto *n, auto *e) {
          return aclnnScatterNdUpdateGetWorkspaceSize(kc, indices, kt, n, e);
        },
        "append paged K");
    b.step(
        aclnnScatterNdUpdate,
        [&](auto *n, auto *e) {
          return aclnnScatterNdUpdateGetWorkspaceSize(vc, indices, vt, n, e);
        },
        "append paged V");
    auto *lookup = b.tensor(s->region(table, 0, batch * context * 8),
                            {batch, context}, ACL_INT64);
    void *kp = b.temp(batch * context * 512 * 2);
    void *vp = b.temp(batch * context * 512 * 2);
    auto *gk = b.tensor(kp, {batch, context, 512}, ACL_BF16);
    auto *gv = b.tensor(vp, {batch, context, 512}, ACL_BF16);
    b.step(
        aclnnEmbedding,
        [&](auto *n, auto *e) {
          return aclnnEmbeddingGetWorkspaceSize(kc, lookup, gk, n, e);
        },
        "gather paged K");
    b.step(
        aclnnEmbedding,
        [&](auto *n, auto *e) {
          return aclnnEmbeddingGetWorkspaceSize(vc, lookup, gv, n, e);
        },
        "gather paged V");
    auto *km =
        b.tensor(kp, {batch, 2, 256, context}, ACL_BF16,
                 {context * 512, 256, 1, 512}, 0, {batch, context, 2, 256});
    auto *vm =
        b.tensor(vp, {batch, 2, context, 256}, ACL_BF16,
                 {context * 512, 256, 512, 1}, 0, {batch, context, 2, 256});
    auto *scores = b.scratch({batch, 2, 4, context}, ACL_BF16);
    b.matmul(qt, km, scores);
    auto *scaled = b.scratch({batch, 2, 4, context}, ACL_BF16);
    auto *scale = b.scalar(0.0625f);
    b.step(
        aclnnMuls,
        [&](auto *n, auto *e) {
          return aclnnMulsGetWorkspaceSize(scores, scale, scaled, n, e);
        },
        "attention score scale");
    auto *sf = b.cast(scaled, {batch, 2, 4, context}, ACL_FLOAT);
    auto *mt = b.tensor(s->region(mask, 0, batch * context * 4),
                        {batch, 1, 1, context}, ACL_FLOAT);
    auto *masked = b.scratch({batch, 2, 4, context});
    b.add(sf, mt, masked);
    auto *prob = b.scratch({batch, 2, 4, context});
    b.step(
        aclnnSoftmax,
        [&](auto *n, auto *e) {
          return aclnnSoftmaxGetWorkspaceSize(masked, -1, prob, n, e);
        },
        "attention FP32 softmax");
    auto *pb = b.cast(prob, {batch, 2, 4, context}, ACL_BF16);
    auto *result = b.scratch({batch, 2, 4, 256}, ACL_BF16);
    b.matmul(pb, vm, result);
    auto *gt = b.tensor(s->region(gate, 0, batch * 2048 * 2),
                        {batch, 2, 4, 256}, ACL_BF16);
    auto *ot = b.tensor(s->region(out, 0, batch * 2048 * 2), {batch, 2, 4, 256},
                        ACL_BF16);
    b.mul(result, gt, ot);
    *operation = b.finish();
  });
}

int32_t pangu_acl_embedding_prepare(pangu_acl_session *s, uint64_t weight,
                                    uint64_t ids, uint64_t out, int64_t vocab,
                                    int64_t width, uint64_t *operation) {
  return boundary([&] {
    if (!s || !operation || vocab <= 0 || vocab > 1048576 || width <= 0 ||
        width > 16384)
      throw std::runtime_error("invalid embedding dimensions");
    Composite b(s);
    auto *w = b.tensor(s->region(weight, 0, vocab * width * 2), {vocab, width},
                       ACL_BF16);
    auto *i = b.tensor(s->region(ids, 0, 8), {1}, ACL_INT64);
    auto *y = b.tensor(s->region(out, 0, width * 2), {1, width}, ACL_BF16);
    b.step(
        aclnnEmbedding,
        [&](auto *n, auto *e) {
          return aclnnEmbeddingGetWorkspaceSize(w, i, y, n, e);
        },
        "token embedding");
    *operation = b.finish();
  });
}
int32_t pangu_acl_model_layer_prepare(
    pangu_acl_session *s, int32_t kind, const uint64_t *w, uint64_t count,
    uint64_t input, uint64_t output, uint64_t positions, uint64_t append,
    uint64_t table, uint64_t mask, int64_t slots, int64_t context,
    uint64_t *state_a, uint64_t *state_b, uint64_t *operation) {
  return boundary([&] {
    if (!s || !w || !state_a || !state_b || !operation ||
        !((kind == 0 && count == 14) || (kind == 1 && count == 11)))
      throw std::runtime_error("invalid model layer bindings");
    s->enter();
    std::vector<uint64_t> ops;
    auto checked = [](int status) {
      if (status)
        throw std::runtime_error(last_error);
    };
    auto alloc = [&](uint64_t bytes) {
      uint64_t id = 0;
      checked(pangu_acl_allocate(s, bytes, &id));
      return id;
    };
    auto emit = [&](auto build) {
      uint64_t id = 0;
      checked(build(&id));
      ops.push_back(id);
    };
    auto linear = [&](uint64_t x, uint64_t weight, uint64_t y, int64_t n,
                      int64_t k) {
      emit([&](auto *id) {
        return pangu_acl_linear_prepare(s, x, weight, y, 1, n, k, id);
      });
    };
    auto point = [&](uint64_t a, uint64_t b, uint64_t y, int64_t n,
                     int32_t mode) {
      emit([&](auto *id) {
        return pangu_acl_pointwise_prepare(s, a, b, y, n, mode, id);
      });
    };
    auto normed = alloc(2048 * 2);
    emit([&](auto *id) {
      return pangu_acl_rms_prepare(s, input, w[0], normed, 1, 2048, id);
    });
    auto attn = alloc(2048 * 2);
    if (kind == 0) {
      auto qkv = alloc(6144 * 2), z = alloc(2048 * 2), a = alloc(16 * 2),
           b = alloc(16 * 2), conv = alloc(6144 * 2);
      auto q = alloc(2048 * 2), k = alloc(2048 * 2), v = alloc(2048 * 2),
           g = alloc(16 * 4), beta = alloc(16 * 2), core = alloc(2048 * 2),
           gated = alloc(2048 * 2);
      *state_a = alloc(6144 * 3 * 2);
      *state_b = alloc(16 * 128 * 128 * 4);
      linear(normed, w[5], qkv, 6144, 2048);
      linear(normed, w[6], z, 2048, 2048);
      linear(normed, w[7], a, 16, 2048);
      linear(normed, w[8], b, 16, 2048);
      emit([&](auto *id) {
        return pangu_acl_conv_prepare(s, qkv, w[9], *state_a, conv, 1, 6144,
                                      id);
      });
      emit([&](auto *id) {
        return pangu_acl_delta_transforms_prepare(s, conv, a, b, w[10], w[11],
                                                  q, k, v, g, beta, 1, id);
      });
      emit([&](auto *id) {
        return pangu_acl_delta_prepare(s, q, k, v, g, beta, *state_b, core, 16,
                                       128, 128, id);
      });
      emit([&](auto *id) {
        return pangu_acl_gated_rms_prepare(s, core, z, w[12], gated, 16, 128,
                                           id);
      });
      linear(gated, w[13], attn, 2048, 2048);
    } else {
      auto qp = alloc(4096 * 2), ki = alloc(512 * 2), vi = alloc(512 * 2),
           q = alloc(2048 * 2), k = alloc(512 * 2), gate = alloc(2048 * 2),
           att = alloc(2048 * 2);
      *state_a = alloc(slots * 512 * 2);
      *state_b = alloc(slots * 512 * 2);
      linear(normed, w[5], qp, 4096, 2048);
      linear(normed, w[6], ki, 512, 2048);
      linear(normed, w[7], vi, 512, 2048);
      emit([&](auto *id) {
        return pangu_acl_attention_qk_prepare(s, qp, ki, w[9], w[10], positions,
                                              q, k, gate, 1, id);
      });
      emit([&](auto *id) {
        return pangu_acl_paged_attention_prepare(s, q, k, vi, gate, *state_a,
                                                 *state_b, append, table, mask,
                                                 att, 1, slots, context, id);
      });
      linear(att, w[8], attn, 2048, 2048);
    }
    auto residual = alloc(2048 * 2), ff = alloc(2048 * 2),
         gate = alloc(6144 * 2), up = alloc(6144 * 2), act = alloc(6144 * 2),
         product = alloc(6144 * 2), down = alloc(2048 * 2);
    point(input, attn, residual, 2048, 0);
    emit([&](auto *id) {
      return pangu_acl_rms_prepare(s, residual, w[1], ff, 1, 2048, id);
    });
    linear(ff, w[2], gate, 6144, 2048);
    linear(ff, w[3], up, 6144, 2048);
    point(gate, 0, act, 6144, 2);
    point(act, up, product, 6144, 1);
    linear(product, w[4], down, 2048, 6144);
    point(residual, down, output, 2048, 0);
    checked(pangu_acl_sequence_prepare(s, ops.data(), ops.size(), operation));
  });
}
int32_t pangu_acl_embedding_batch_prepare(pangu_acl_session *s, uint64_t weight,
                                          uint64_t ids, uint64_t out,
                                          int64_t vocab, int64_t width,
                                          int64_t batch, uint64_t *operation) {
  return boundary([&] {
    if (!s || !operation || vocab <= 0 || vocab > 1048576 || width <= 0 ||
        width > 16384 || batch < 1 || batch > 64)
      throw std::runtime_error("invalid embedding dimensions");
    Composite b(s);
    auto *w = b.tensor(s->region(weight, 0, vocab * width * 2), {vocab, width},
                       ACL_BF16);
    auto *i = b.tensor(s->region(ids, 0, batch * 8), {batch}, ACL_INT64);
    auto *y = b.tensor(s->region(out, 0, batch * width * 2), {batch, width},
                       ACL_BF16);
    b.step(
        aclnnEmbedding,
        [&](auto *n, auto *e) {
          return aclnnEmbeddingGetWorkspaceSize(w, i, y, n, e);
        },
        "token embedding");
    *operation = b.finish();
  });
}
int32_t pangu_acl_model_layer_batch_prepare(
    pangu_acl_session *s, int32_t kind, const uint64_t *w, uint64_t count,
    uint64_t input, uint64_t output, uint64_t positions, uint64_t append,
    uint64_t table, uint64_t mask, int64_t slots, int64_t context,
    int64_t batch, uint64_t active, uint64_t *state_a, uint64_t *state_b,
    uint64_t *operation) {
  return boundary([&] {
    if (batch < 1 || batch > 64 || !s || !w || !state_a || !state_b ||
        !operation ||
        !((kind == 0 && count == 14) || (kind == 1 && count == 11)))
      throw std::runtime_error("invalid model layer bindings");
    s->enter();
    std::vector<uint64_t> ops;
    auto checked = [](int status) {
      if (status)
        throw std::runtime_error(last_error);
    };
    auto alloc = [&](uint64_t bytes) {
      uint64_t id = 0;
      checked(pangu_acl_allocate(s, bytes * batch, &id));
      return id;
    };
    auto emit = [&](auto build) {
      uint64_t id = 0;
      checked(build(&id));
      ops.push_back(id);
    };
    auto linear = [&](uint64_t x, uint64_t weight, uint64_t y, int64_t n,
                      int64_t k) {
      emit([&](auto *id) {
        return pangu_acl_linear_prepare(s, x, weight, y, batch, n, k, id);
      });
    };
    auto point = [&](uint64_t a, uint64_t b, uint64_t y, int64_t n,
                     int32_t mode) {
      emit([&](auto *id) {
        return pangu_acl_pointwise_prepare(s, a, b, y, batch * n, mode, id);
      });
    };
    std::vector<uint64_t> restore_ops;
    auto preserve = [&](uint64_t state, int64_t width, aclDataType dtype) {
      Composite before(s), after(s);
      auto bytes = uint64_t(batch * width * (dtype == ACL_FLOAT ? 4 : 2));
      auto *saved_ptr = before.temp(bytes);
      before.op->copies.push_back(
          {saved_ptr, s->region(state, 0, bytes), bytes});
      ops.push_back(before.finish());
      auto *current =
          after.tensor(s->region(state, 0, bytes), {batch, width}, dtype);
      auto *old = after.tensor(saved_ptr, {batch, width}, dtype);
      auto *condition =
          after.tensor(s->region(active, 0, batch), {batch, 1}, ACL_BOOL);
      auto *selected_ptr = after.temp(bytes);
      auto *selected = after.tensor(selected_ptr, {batch, width}, dtype);
      after.step(
          aclnnSWhere,
          [&](auto *n, auto *e) {
            return aclnnSWhereGetWorkspaceSize(condition, current, old,
                                               selected, n, e);
          },
          "preserve inactive recurrent lanes");
      after.op->copies.push_back(
          {s->region(state, 0, bytes), selected_ptr, bytes});
      restore_ops.push_back(after.finish());
    };
    auto normed = alloc(2048 * 2);
    emit([&](auto *id) {
      return pangu_acl_rms_prepare(s, input, w[0], normed, batch, 2048, id);
    });
    auto attn = alloc(2048 * 2);
    if (kind == 0) {
      auto qkv = alloc(6144 * 2), z = alloc(2048 * 2), a = alloc(16 * 2),
           b = alloc(16 * 2), conv = alloc(6144 * 2);
      auto q = alloc(2048 * 2), k = alloc(2048 * 2), v = alloc(2048 * 2),
           g = alloc(16 * 4), beta = alloc(16 * 2), core = alloc(2048 * 2),
           gated = alloc(2048 * 2);
      *state_a = alloc(6144 * 3 * 2);
      *state_b = alloc(16 * 128 * 128 * 4);
      preserve(*state_a, 6144 * 3, ACL_BF16);
      preserve(*state_b, 16 * 128 * 128, ACL_FLOAT);
      linear(normed, w[5], qkv, 6144, 2048);
      linear(normed, w[6], z, 2048, 2048);
      linear(normed, w[7], a, 16, 2048);
      linear(normed, w[8], b, 16, 2048);
      emit([&](auto *id) {
        return pangu_acl_conv_prepare(s, qkv, w[9], *state_a, conv, batch, 6144,
                                      id);
      });
      emit([&](auto *id) {
        return pangu_acl_delta_transforms_prepare(s, conv, a, b, w[10], w[11],
                                                  q, k, v, g, beta, batch, id);
      });
      emit([&](auto *id) {
        return pangu_acl_delta_prepare(s, q, k, v, g, beta, *state_b, core,
                                       batch * 16, 128, 128, id);
      });
      emit([&](auto *id) {
        return pangu_acl_gated_rms_prepare(s, core, z, w[12], gated, batch * 16,
                                           128, id);
      });
      linear(gated, w[13], attn, 2048, 2048);
    } else {
      auto qp = alloc(4096 * 2), ki = alloc(512 * 2), vi = alloc(512 * 2),
           q = alloc(2048 * 2), k = alloc(512 * 2), gate = alloc(2048 * 2),
           att = alloc(2048 * 2);
      *state_a = alloc((slots / batch) * 512 * 2);
      *state_b = alloc((slots / batch) * 512 * 2);
      linear(normed, w[5], qp, 4096, 2048);
      linear(normed, w[6], ki, 512, 2048);
      linear(normed, w[7], vi, 512, 2048);
      emit([&](auto *id) {
        return pangu_acl_attention_qk_prepare(s, qp, ki, w[9], w[10], positions,
                                              q, k, gate, batch, id);
      });
      emit([&](auto *id) {
        return pangu_acl_paged_attention_prepare(
            s, q, k, vi, gate, *state_a, *state_b, append, table, mask, att,
            batch, slots, context, id);
      });
      linear(att, w[8], attn, 2048, 2048);
    }
    auto residual = alloc(2048 * 2), ff = alloc(2048 * 2),
         gate = alloc(6144 * 2), up = alloc(6144 * 2), act = alloc(6144 * 2),
         product = alloc(6144 * 2), down = alloc(2048 * 2);
    point(input, attn, residual, 2048, 0);
    emit([&](auto *id) {
      return pangu_acl_rms_prepare(s, residual, w[1], ff, batch, 2048, id);
    });
    linear(ff, w[2], gate, 6144, 2048);
    linear(ff, w[3], up, 6144, 2048);
    point(gate, 0, act, 6144, 2);
    point(act, up, product, 6144, 1);
    linear(product, w[4], down, 2048, 6144);
    point(residual, down, output, 2048, 0);
    ops.insert(ops.end(), restore_ops.begin(), restore_ops.end());
    checked(pangu_acl_sequence_prepare(s, ops.data(), ops.size(), operation));
  });
}
int32_t pangu_acl_copy_region(pangu_acl_session *s, uint64_t src,
                              uint64_t src_offset, uint64_t dst,
                              uint64_t dst_offset, uint64_t bytes) {
  return boundary([&] {
    if (!s)
      throw std::runtime_error("null copy session");
    s->enter();
    check(aclrtMemcpyAsync(s->region(dst, dst_offset, bytes), bytes,
                           s->region(src, src_offset, bytes), bytes,
                           ACL_MEMCPY_DEVICE_TO_DEVICE, s->stream),
          "copy region");
    check(aclrtSynchronizeStream(s->stream), "copy region fence");
  });
}
int32_t pangu_acl_close(pangu_acl_session *s) {
  return boundary([&] {
    if (!s)
      return;
    if (s->thread != std::this_thread::get_id())
      throw std::runtime_error("wrong close thread");
    if (s->context)
      check(aclrtSetCurrentContext(s->context), "close context");
    if (s->stream)
      check(aclrtSynchronizeStream(s->stream), "close fence");
    while (!s->graphs.empty()) {
      auto it = s->graphs.begin();
      check(aclmdlRIDestroy(it->second), "graph destroy");
      s->graphs.erase(it);
    }
    s->operations.clear();
    if (s->communicator) {
      auto result = HcclCommDestroy(s->communicator);
      if (result != HCCL_SUCCESS)
        throw std::runtime_error("TP destroy: " + std::to_string(result));
      s->communicator = nullptr;
    }
    while (!s->buffers.empty()) {
      auto it = s->buffers.begin();
      check(aclrtFree(it->second.ptr), "buffer free");
      s->buffers.erase(it);
    }
    if (s->stream) {
      check(aclrtDestroyStream(s->stream), "stream destroy");
      s->stream = nullptr;
    }
    if (s->context) {
      check(aclrtDestroyContext(s->context), "context destroy");
      s->context = nullptr;
    }
    std::lock_guard<std::mutex> guard(init_mutex);
    if (init_users == 1)
      check(aclFinalize(), "aclFinalize");
    --init_users;
    delete s;
  });
}
}

#include "sampler.inc"
