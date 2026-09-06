# Native implementation boundary

Intentionally no CANN headers, FFI declarations, build scripts or kernel source yet.

The NPU-node agent owns the vendor compiler adapter, checked C ABI bridge, ACL/ACLNN/HCCL/HIXL calls and Ascend C templates. Follow [the handoff contract](../docs/npu-handoff.md). CPU-only `cargo test --workspace` must remain independent of CANN and Python.
