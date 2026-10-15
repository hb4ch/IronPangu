#pragma once
#include "inferfabric_kernel_v1.h"
#include <memory>
#include <string>
#include <vector>
namespace inferfabric {
struct KernelModule;
class PreparedKernel {
  friend class CompiledKernel;
  std::shared_ptr<KernelModule> module_;
  IFKernelPreparedV1 prepared_{};
  mutable bool enqueued_ = false;
  PreparedKernel(std::shared_ptr<KernelModule>, IFKernelPreparedV1);

public:
  PreparedKernel(const PreparedKernel &) = delete;
  PreparedKernel &operator=(const PreparedKernel &) = delete;
  PreparedKernel(PreparedKernel &&) noexcept;
  ~PreparedKernel();
  uint64_t workspace_bytes() const { return prepared_.workspace_bytes; }
  void enqueue(void *workspace, uint64_t capacity, void *stream) const;
};
class CompiledKernel {
  std::shared_ptr<KernelModule> module_;

public:
  // Explicit trusted native-code selection, never initiated by parsing a DSL.
  CompiledKernel(const std::string &library, const std::string &contract,
                 const std::string &target);
  PreparedKernel prepare(const std::vector<IFKernelTensorV1> &inputs,
                         const std::vector<IFKernelTensorV1> &outputs) const;
};
} // namespace inferfabric
