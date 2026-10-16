#pragma once
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace inferfabric::initialization {
using ParameterId = uint32_t;
enum class DType : uint32_t { I8 = 1, Bf16 = 2, F32 = 3 };
uint64_t element_bytes(DType dtype);
struct Parameter {
  ParameterId id;
  std::string
      name; // Debug/binding metadata, never interpreted by the executor.
  DType dtype;
  std::vector<int64_t> shape;
  uint64_t bytes() const;
};
struct LoadSlice {
  ParameterId destination;
  std::string shard;
  uint64_t offset, bytes;
  DType source_dtype;
  uint64_t row_bytes = 0, stride_bytes = 0;
  uint64_t source_span() const;
};
struct Plan {
  uint32_t rank, device;
  std::string
      checkpoint_metadata_sha256; // Provenance, not a payload integrity check.
  std::vector<Parameter> parameters;
  std::vector<LoadSlice> instructions;
  static Plan read(const std::string &path);
  void validate() const;
  ParameterId resolve(const std::string &name) const;
};

// Backend operations are synchronous: upload completion precedes staging reuse.
// Allocation owners retain the backend until their final release.
class DeviceMemory {
public:
  virtual ~DeviceMemory() = default;
  virtual void *allocate(uint64_t bytes) = 0;
  virtual void release(void *pointer) noexcept = 0;
  virtual void upload(void *destination, const void *source, size_t bytes) = 0;
  virtual void readback(void *destination, const void *source,
                        size_t bytes) = 0;
};
class ResidentParameter {
  std::shared_ptr<DeviceMemory> backend_;

public:
  Parameter descriptor;
  void *data = nullptr;
  uint64_t bytes;
  DType dtype;
  ResidentParameter(const Parameter &parameter,
                    std::shared_ptr<DeviceMemory> backend);
  ~ResidentParameter();
  ResidentParameter(const ResidentParameter &) = delete;
  ResidentParameter &operator=(const ResidentParameter &) = delete;
};
class ParameterStore {
  friend class Executor;
  std::vector<std::unique_ptr<ResidentParameter>> parameters_;

public:
  uint64_t source_bytes = 0, resident_bytes = 0;
  ResidentParameter &at(ParameterId id) const;
  size_t size() const { return parameters_.size(); }
};
struct Options {
  size_t staging_bytes = 4 * 1024 * 1024;
  uint64_t resident_budget = std::numeric_limits<uint64_t>::max();
  bool verify_upload = false;
};
class Executor {
public:
  // All-or-nothing publication: validation precedes allocation, and failures
  // release every allocation; a store is returned only after all uploads
  // finish.
  static ParameterStore execute(const Plan &plan, const std::string &root,
                                std::shared_ptr<DeviceMemory> backend,
                                const Options &options = {});
};
} // namespace inferfabric::initialization
