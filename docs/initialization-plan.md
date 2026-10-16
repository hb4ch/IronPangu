# Parameter initialization plans

The executable 92B DSL derives tensor requirements, TP row/column slices and EP ownership directly from graph operators via `inferfabric-compiler::token`. The older OpenPangu placement adapter remains available for audits, but the DSL execution path bypasses its model-name rules. Both emit the permanent initialization IR described here. See [the executable DSL](openpangu-executable-dsl.md).

The shared initialization subsystem is model independent. Model binding/lowering produces `inferfabric_ir::initialization::InitializationPlan`; the compiler encodes it as `IFINIT01`. The native executor knows parameter IDs, storage slices, dtypes and device memory operations, not model names or TP/EP semantics.

## Contract

A plan declares rank/device metadata, a checkpoint metadata fingerprint (provenance only), dense artifact-local parameter IDs, debug/binding names, shapes and destination dtypes. Its ordered initialization program contains `LoadSlice` instructions. Each instruction allocates one declared parameter and streams a contiguous file range into it. Equal dtypes copy bytes; BF16 to FP32 is the only supported conversion. Conversion and upload are fused to avoid allocating a second full tensor. Every parameter must have exactly one producer. Unsupported operations and conversions fail closed.

An optional `source_rows` specifies `row_bytes` and `stride_bytes` for strided
source reads. `bytes` remains the selected packed byte count; the source span is
`(bytes / row_bytes - 1) * stride_bytes + row_bytes`. Validation checks alignment,
divisibility, non-overlap, overflow and the complete file span before allocation.
Rows are packed into bounded staging chunks and uploaded contiguously, without
loading or allocating the unsharded matrix. This is a generic storage operation;
the executor does not interpret tensor names, ranks or model architecture.

`linear(placement="tp_columns")` lowers to these strided reads, with replicated
per-output scales. `mome(placement="tp_channels")` uses contiguous row slicing
for its independent channels. The DSL explicitly supplies the following reduction.

This is an executable initialization IR, not the model execution DAG. Beyond
explicit strided source-row packing, it has no general transpose, gamma folding
or compute-kernel instruction. Such preparation remains explicit runtime work
until represented by validated IR operations; a transform string cannot silently
request unimplemented behavior.

## Wire format

All integers are little endian. Strings are a u32 byte length followed by UTF-8 bytes (1..4096, no NUL). The native backend currently targets little-endian systems.

- Eight-byte magic `IFINIT01`, u32 rank, u32 device, fingerprint string.
- u32 parameter count. For each: u32 ID, name string, u32 dtype, u32 dimension count, u64 dimensions.
- u32 instruction count. For each: u32 opcode, u32 destination ID, shard string, u64 absolute file offset, u64 source bytes, u32 source dtype.
- Dtypes: I8=1, BF16=2, F32=3. Opcode 1 is `LoadSlice`.
- Opcode 2 has the same fields followed by u64 row bytes and u64 source stride.
  Opcode-1 artifacts remain byte-identical; older readers reject opcode 2.

Counts are bounded at 100,000, dimensions at eight. IDs are dense and positional; instruction order may differ from ID order. Validation rejects unknown opcodes/dtypes, duplicate names/producers, missing parameters, malformed paths, invalid shape/range arithmetic and trailing bytes. `IFLOAD01` is an obsolete development format and is rejected, not guessed or silently converted.

## Execution and ownership

`native/include/initialization_plan.h` defines the plan, typed parameter store and synchronous `DeviceMemory` interface. `initialization_plan.cpp` validates and executes it without ACL dependencies. `initialization_acl.cpp` provides the Ascend allocation/upload/readback adapter. The caller owns the device context and must retain it until parameters are released.

All plans and file ranges are checked before device allocation. A resident-byte budget can reject the plan before allocation. File input is staged in 4 MiB chunks by default, with at most 8 MiB conversion scratch and optional 8 MiB verification scratch. Uploads finish before buffers are reused. A failed read, allocation, conversion or upload unwinds all allocations; a parameter store is published only after complete success. Device allocations retain their backend ownership, and are addressed by parameter ID. Names are binding/debug metadata.

Storage must remain immutable during initialization. The metadata fingerprint is not a cryptographic payload check. File paths/ranges are checked, but replacement/truncation races and authenticated content storage are not solved by this format. Device memory accounting here covers resident parameters only, not activations, workspaces or other runtime preparation.

## Validation

Rust IR tests cover generic ranks, shape/ID/path/conversion checks. The native backend-independent test exercises multi-chunk copies and casts, exact readback, budget rejection, source bounds, producer validation, nonfinite cast rejection and injected upload failure with zero remaining allocations. Seventeen wire-format cases cover malformed binary inputs. Build native tests inside Docker:

```bash
cmake -S native -B native-build -DCMAKE_BUILD_TYPE=Release
cmake --build native-build --target initialization_test inferfabric_initialization_acl -j4
native-build/initialization_test
python3 native/probes/test_initialization_plan.py native-build/initialization_test
```

## OpenPangu integration

`openpangu::parallel::initialization_plan` translates model placement decisions into this shared IR. The exporter delegates binary serialization to `inferfabric_compiler::initialization::encode`; it contains no private wire encoder. Every rank also receives a readable JSON dump of the exact initialization IR. The decoder binds model names to artifact parameter IDs once at startup and retrieves allocations through `ParameterStore::at`. The generic executor never interprets names or expert ownership. The old `openpangu_weights.h`, model-specific loader probe and IFLOAD01 parser are removed.

The decoder's own operation selection and name-to-ID binding table remain model-specific. mHC gamma folding, router transposition and MoME layout preparation are still explicit operator preparation outside this first initialization instruction set. They are not silently encoded as no-op initialization transforms. Moving those into typed preparation instructions is separate from migrating file I/O, residency and conversions onto this shared executor.
