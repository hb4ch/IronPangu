#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// POD-only ABI. No exceptions, STL objects or ownership transfers cross it.
// Dtypes follow initialization IR: I8=1, BF16=2, F32=3.
typedef struct {
  uint32_t struct_bytes, dtype, rank, access; // access: read=1, write=2
  void *data;
  uint64_t allocation_bytes;
  int64_t shape[8], strides[8]; // Strides in elements.
} IFKernelTensorV1;
typedef struct {
  void *state;
  uint64_t workspace_bytes, workspace_alignment;
} IFKernelPreparedV1;
typedef struct {
  uint32_t struct_bytes, abi_version;
  const char *contract;
  const char *target;
  // Prepare may allocate metadata, but must not launch device work. Buffers
  // remain fixed through enqueue/completion. On failure, out must be empty.
  int32_t (*prepare)(const IFKernelTensorV1 *inputs, uint32_t input_count,
                     const IFKernelTensorV1 *outputs, uint32_t output_count,
                     IFKernelPreparedV1 *out);
  // No allocations, compilation, synchronization or hidden stream changes.
  int32_t (*enqueue)(void *state, void *workspace, uint64_t bytes,
                     void *stream);
  // Caller must fence completion before release, including failed enqueue.
  void (*release)(void *state);
} IFKernelV1;
typedef const IFKernelV1 *(*IFKernelEntryV1)(uint32_t requested_version);
// Shared objects export: const IFKernelV1 *inferfabric_kernel_v1(uint32_t).
#ifdef __cplusplus
}
#endif
