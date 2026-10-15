#include "compiled_kernel.h"
#include <dlfcn.h>
#include <limits>
#include <stdexcept>
#include <utility>
namespace inferfabric {
struct KernelModule {
  void *handle = nullptr;
  const IFKernelV1 *api = nullptr;
  ~KernelModule() {
    if (handle)
      dlclose(handle);
  }
};
namespace {
void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
void validate(const IFKernelTensorV1 &t, uint32_t access) {
  require(t.struct_bytes == sizeof(t) && t.data && t.rank > 0 && t.rank <= 8 &&
              t.access == access,
          "kernel tensor descriptor");
  uint64_t bytes = t.dtype == 1 ? 1 : t.dtype == 2 ? 2 : t.dtype == 3 ? 4 : 0;
  require(bytes != 0, "kernel dtype");
  uint64_t stride = 1;
  for (uint32_t i = t.rank; i-- > 0;) {
    require(t.shape[i] > 0 && stride <= INT64_MAX &&
                t.strides[i] == static_cast<int64_t>(stride),
            "kernel requires contiguous positive tensor");
    require(uint64_t(t.shape[i]) <= UINT64_MAX / stride,
            "kernel shape overflow");
    stride *= uint64_t(t.shape[i]);
  }
  require(stride <= UINT64_MAX / bytes && stride * bytes <= t.allocation_bytes,
          "kernel tensor allocation bound");
}
} // namespace
CompiledKernel::CompiledKernel(const std::string &library,
                               const std::string &contract,
                               const std::string &target)
    : module_(std::make_shared<KernelModule>()) {
  module_->handle = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!module_->handle)
    throw std::runtime_error(dlerror());
  auto entry = reinterpret_cast<IFKernelEntryV1>(
      dlsym(module_->handle, "inferfabric_kernel_v1"));
  require(entry != nullptr, "native kernel entry missing");
  module_->api = entry(1);
  auto api = module_->api;
  require(api && api->struct_bytes == sizeof(IFKernelV1) &&
              api->abi_version == 1,
          "native kernel ABI mismatch");
  require(api->contract && api->target && contract == api->contract &&
              target == api->target,
          "native kernel contract/target mismatch");
  require(api->prepare && api->enqueue && api->release,
          "native kernel function table incomplete");
}
PreparedKernel
CompiledKernel::prepare(const std::vector<IFKernelTensorV1> &inputs,
                        const std::vector<IFKernelTensorV1> &outputs) const {
  require(!inputs.empty() && !outputs.empty() && inputs.size() <= 64 &&
              outputs.size() <= 64,
          "kernel port count");
  for (const auto &v : inputs)
    validate(v, 1);
  for (const auto &v : outputs)
    validate(v, 2);
  // Initial ABI supports disjoint contiguous outputs, not in-place aliases.
  auto overlaps = [](const IFKernelTensorV1 &a, const IFKernelTensorV1 &b) {
    auto x = reinterpret_cast<uintptr_t>(a.data),
         y = reinterpret_cast<uintptr_t>(b.data);
    require(a.allocation_bytes <= UINTPTR_MAX - x &&
                b.allocation_bytes <= UINTPTR_MAX - y,
            "kernel address overflow");
    return x < y + b.allocation_bytes && y < x + a.allocation_bytes;
  };
  for (size_t i = 0; i < outputs.size(); ++i) {
    for (const auto &input : inputs)
      require(!overlaps(outputs[i], input), "kernel output aliases input");
    for (size_t j = 0; j < i; ++j)
      require(!overlaps(outputs[i], outputs[j]), "kernel outputs alias");
  }
  IFKernelPreparedV1 p{};
  int status = module_->api->prepare(inputs.data(), inputs.size(),
                                     outputs.data(), outputs.size(), &p);
  if (status != 0) {
    if (p.state)
      module_->api->release(p.state);
    throw std::runtime_error("kernel prepare failed: " +
                             std::to_string(status));
  }
  PreparedKernel result(module_, p);
  require(p.state && p.workspace_alignment &&
              !(p.workspace_alignment & (p.workspace_alignment - 1)) &&
              p.workspace_alignment <= 4096,
          "kernel workspace contract");
  return result;
}
PreparedKernel::PreparedKernel(std::shared_ptr<KernelModule> module,
                               IFKernelPreparedV1 p)
    : module_(std::move(module)), prepared_(p) {}
PreparedKernel::PreparedKernel(PreparedKernel &&other) noexcept
    : module_(std::move(other.module_)),
      prepared_(std::exchange(other.prepared_, {})),
      enqueued_(std::exchange(other.enqueued_, true)) {}
PreparedKernel::~PreparedKernel() {
  if (prepared_.state)
    module_->api->release(prepared_.state);
}
void PreparedKernel::enqueue(void *workspace, uint64_t capacity,
                             void *stream) const {
  require(capacity >= prepared_.workspace_bytes, "kernel workspace capacity");
  if (prepared_.workspace_bytes)
    require(workspace && reinterpret_cast<uintptr_t>(workspace) %
                                 prepared_.workspace_alignment ==
                             0,
            "kernel workspace alignment");
  require(!enqueued_,
          "prepared kernel is single-enqueue; prepare a new invocation");
  enqueued_ = true;
  int status =
      module_->api->enqueue(prepared_.state, workspace, capacity, stream);
  if (status != 0)
    throw std::runtime_error("kernel enqueue failed: " +
                             std::to_string(status));
}
} // namespace inferfabric
