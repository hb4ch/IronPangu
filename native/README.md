# Native CANN boundary

`include/inferfabric_acl.h` defines a versioned C ABI implemented by `src/inferfabric_acl.cpp`.
It owns thread-affine ACL contexts/streams, opaque allocation and graph IDs, checked byte regions, synchronous host copies, graph capture/replay, and ordered cleanup. C++ exceptions become error status/text.

Build only inside the remote Docker container:

```sh
cmake -S native -B native-build
cmake --build native-build -j4
./native-build/acl_graph_probe --abi
```

CANN_ROOT defaults to `/usr/local/Ascend/cann`. The build uses installed headers and libraries. CPU Cargo builds have no CANN dependency.

`acl_graph_probe DEVICE_ID` captures a D2D copy and verifies three replays after changing the source buffer. It is a weight-independent qualification probe, not attention metadata or model graph validation. Do not run it on an unassigned device. Device execution and graph replay have been qualified on both exposed logical devices.

Session methods must run on their creating thread. Close fences work and destroys graphs before buffers. Cleanup failure retains the session for retry; callers must not free referenced memory. After capture failure, ordinary operations fail closed. The bridge deliberately exposes no individual buffer release that could invalidate a live graph.

This ABI has not yet been wired into `AscendBackend`. The separate `inferfabric-native` runner now builds a complete model graph with ACLNN math. HCCL, HIXL and scheduler integration remain outstanding. Follow `docs/npu-handoff.md` and `docs/remote-development.md`.

Native ABI version 2 now includes prepared BF16 linear and weighted RMS operations, tested against real checkpoint weights on two devices. See [the NPU bring-up report](../docs/npu-bringup.md) for executor retention, graph lifetime, build commands, numerical scope and remaining work.

See [the native milestone](../docs/npu-native-milestone.md) for full-model and HTTP qualification, FP32 recurrence, convolution, paged attention and current serving limits.

`physical_plan_probe FIXTURE DEVICE` is an explicit hardware qualification executable for trusted fixtures emitted by `cargo run -p inferfabric-cli --example export_npu_fixture`. It tests planned arena reuse and state commits through ACLNN eager/captured execution. It is not a production plan loader or new serving backend. [Two-device qualification and reproduction](../docs/validation/2026-09-08-npu-planning/README.md).

## Compiled operation implementations

`inferfabric_kernel_v1.h` defines a POD C ABI for separately compiled operations. `CompiledKernel` owns a shared-library handle, validates ABI/contract/target and contiguous tensor descriptors, and creates single-enqueue prepared invocations with caller-owned workspace. The 92B decoder's dense/shared BF16 SwiGLU uses this interface. Default implementation: `kernels/swiglu_cann.cpp`; override: `INFERFABRIC_SWIGLU_LIBRARY=/absolute/path/library.so`. A compiled Ascend C host launcher may implement the same contract. See [authoring and implementation boundaries](../docs/model-authoring.md#native-ascend-c-boundary).

Build/check the ABI without model loading inside Docker:

```bash
cmake --build native-build --target compiled_kernel_test kernel_test_impl kernel_bad_impl -j4
native-build/compiled_kernel_test native-build/libkernel_test_impl.so native-build/libkernel_bad_impl.so
```

These interface tests do not qualify arbitrary plugin code or capture safety. Use the complete 92B generation comparison after changing its selected implementation. The qualifier snapshots/hashes both the executable and selected library. Kernel input/output buffers and prepared state must remain alive until stream completion.
