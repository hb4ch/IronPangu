# Native physical planning and device-buffer execution

Design accepted 2026-10-16. This extends [physical planning](physical-planning.md)
from the CPU prototype to the native token program. The existing CPU v1 planner
and executor remain supported; new target-independent contracts live in
`inferfabric-plan`, with backend lowering in `inferfabric-compiler`.

## Objective and boundary

DSL or compiled-language authors produce the same validated logical graph.
Planning chooses implementations, layouts, storage and per-rank schedules.
Execution binds resources and executes the frozen choices. It must not recover
architecture from names, silently substitute kernels or rerun optimization.

```text
expanded logical DAG + target + shape/topology constraints
    -> legal implementation/layout candidates
    -> cost-guided selection and per-rank schedule
    <-> activation/workspace allocation and resource reconciliation
    -> verification -> physical bundle
    -> prepare process-local resources -> execute bounded invocations
```

Kernel calls may lower one logical node into many launches, or fuse several
nodes into one implementation. Preserve logical/source identities throughout.
Generation length is dynamic; plans describe bounded invocations, not unrolled
whole conversations. The Rust serving scheduler chooses an eligible shape variant;
physical scheduling controls work inside that invocation.

## Mandatory device-residency invariant — 2026-10-16

Activation tensors, routing IDs/scores, expert outputs, collective buffers and
KV/recurrent state remain on the device for the entire invocation. There is no
host fallback option, cost tradeoff, diagnostic activation-download flag or
implicit offloading. Future KV offloading requires a separate explicit contract.
The physical selector rejects every host/opaque candidate and unclassified
host-transfer boundary **before** reading costs, even an ineligible candidate.
All intermediate value locations must be device-resident.

The native `resident::Tensor` owns shared device storage and exposes no host
iterators/read/upload API. Device copies, casts, arithmetic, rotary application,
routing selection, expert reduction, greedy argmax and output log-softmax execute
on Ascend. Dispatch still follows the generic expanded token program, including
source-defined modules, dependencies and weight bindings. Unsupported operators
or layouts fail validation; there is no alternative host implementation.

Explicit transfer roles are limited to:

- **Control input:** token-position/sequence metadata, static indices and rotary
  coefficients computed from positions. No device activation is downloaded to
  produce these inputs.
- **Parameter preparation:** one-time transformations of immutable checkpoint
  weights (mHC folding, convolution layout, router transpose), currently using
  host C++ preparation. These are parameters, never graph activations.
- **Request output:** selected token consensus IDs (64 bytes per rank per output
  step) and top-five IDs/logits/log-probabilities (80 extra bytes on rank zero).
  Full vocabulary logits remain on device. Prefill exports none of these outputs.

`resident_transfer.h` guards the executor translation unit's synchronous and
asynchronous ACL host copies. Unclassified copies throw; scoped explicit roles
record counts/bytes. `INFERFABRIC_TRANSFER_AUDIT_DIR` writes per-rank JSON. The
qualification driver verifies every rank's ledger and exact output-byte bounds.
This covers executor-owned transfers, not undocumented internal vendor-library
control traffic. Initial checkpoint ingestion is handled separately by the
permanent initialization executor.

The first resident EP implementation packs owner-local expert weights on device,
selects weights using device route IDs, and executes eight fixed route slots per
rank. Foreign-owner slots compute then receive zero device-side coefficients.
This preserves route-order FP32 reduction without exporting routing data, but
adds duplicate weight-bank memory and redundant expert compute. It is a
correctness baseline, not sparse production performance. Grouped sparse dispatch
and packing directly during initialization are subsequent optimization work.

## Current milestone 1

`inferfabric-plan::physical` supplies versioned, backend-neutral inspection IR,
a deterministic legal candidate selector and a serial schedule verifier. The
Ascend token adapter registers the current operator implementations without model
names or layer schedules. `compile-token` publishes `physical.json` alongside
`execution.html`; the viewer embeds the same verified physical metadata.

The resident 92B path uses device values, internal device allocations and frequent
synchronization. The physical artifact remains **inspection-only and non-executable**
until the native executor consumes frozen physical invocations directly. One physical step represents
a backend call with opaque internals, not a claim that each logical op is one
NPU kernel. Boundary annotations identify classified external transfers, synchronization,
allocation and persistent-state handling. They do not pretend to be a complete
internal launch DAG. The native runtime continues consuming `execution.bin`.

Actual decisions now recorded:

- Selected implementation, contract/source fingerprint, numerical contract,
  legal alternatives, rejection reasons and selection rationale.
- Explicit rank/device bindings to shared immutable schedule templates. Identical
  schedules are stored once; the rank selector does not imply measured rank times.
- Data dependencies, serial-order dependencies, stream zero and collective
  group/signature/sequence. Current lowering preserves all-rank collective order.
- Known device output payload shapes/dtypes, inclusive last-use positions, retained
  outputs and zero intermediate host payload. Opaque handles, container overhead,
  device workspace, persistent allocations and HBM totals remain unresolved.
- No planned aliasing/reuse, no stream overlap and no qualified capture regions.
- Unknown costs explicitly represented as unknown, never zero latency.

Source fingerprints identify the adapter and inspected backend sources, not an
installed native binary or vendor library. Deployment identities remain unresolved.
A standalone saved token graph uses the supported eight-rank baseline; compiling a
bundle uses its validated explicit device bindings. The viewer is model-independent.

The generic serial core accepts other frontend/backend contracts. The current
verifier intentionally accepts only the conservative no-reuse, shared serial
schedule policy; broader executable schedules need new validation, not bypasses.

## Physical records and eventual executable bundle

| Record | Required information |
|---|---|
| Tensor/view | Logical origin, shape/bounds, dtype, layout/strides, ownership and aliases |
| Invocation | Implementation identity, numerical contract, specialization, inputs/outputs and workspace |
| Transfer/collective | Buffers, direction, participants, communicator and matching sequence |
| Schedule | Rank, stream, dependency/event edges, conditions and invocation boundaries |
| Allocation | Memory class, arena, offset, bytes, alignment and completion-based lifetimes |
| Decision | Alternatives, estimates, measured evidence, rejected options and constraints |

Separate weights/transformed constants, persistent state, inputs/retained outputs,
activations, workspace, communication/staging buffers and runtime/capture reserves.
Unknown resource requirements prevent an executable plan. No offsets, arena size
or latency should be inferred merely to populate a viewer.

Pointers, communicator handles, prepared ACL objects and graph handles are local
runtime resources. They never enter serialized plans. Kernel binaries and resource
contracts must be pinned before publication. If preparation exceeds a workspace
bound or encounters an incompatible target, fail loading and request explicit
replanning outside the executor.

## Cost model and hardware evidence

First compare legal implementations only: shape/layout/dtype support, numerical
contract, alignment, state effects and capture requirements are hard constraints.
Initial native lowering has one known baseline per op. The generic selector can
compare supplied estimates; it uses the smallest conservative duration bound and
stable implementation-ID tie breaks. If any legal candidate lacks a cost, retain
a deterministic baseline rather than treating missing cost as zero. Current cost
selection tests use synthetic fixtures, not invented hardware measurements.

Cost sources: analytical, measured, interpolated, or unknown. Estimates carry a
pinned evidence/model identity, estimate and upper bound. The complete future
cost database additionally records SoC, topology, clocks/environment, CANN/driver
and implementation hashes, shape/layout/dtype, warmup, sampling distribution,
variability, workspace, eager/captured mode and correctness qualification.

Initial analytical compute estimate:

```text
launch overhead + max(work / effective throughput, traffic / effective bandwidth)
```

Copies and collectives use separate models. Global optimization minimizes
estimated critical-path completion across ranks subject to per-rank memory,
correctness and execution constraints. Retain a small latency/memory Pareto set.
Do not sum overlapped kernel times or assume isolated microbenchmark latency
survives resource contention. Calibrate with region and complete invocation
measurements; report held-out prediction errors and uncertainty.

Hardware sampling is an explicit Docker-native tool, never an implicit compiler
side effect. Rust planning/cross-compilation remains local. Compilation consumes
an immutable cost snapshot. Graph, target, constraints, kernel catalog and cost
snapshot determine the plan reproducibly. Actual NPU cost probes and the global
cost search are later milestones, not implemented by the initial selector.

## Memory reuse and scheduling

Start with device-resident serial execution, then inclusive lifetime allocation.
For multiple streams, a reuse proof requires all prior readers/writers to complete
before a new writer begins; launch order or estimated timestamps are insufficient.
Insert reuse edges only if legal and account for their serialization cost. Re-run
scheduling/storage reconciliation within a bounded search and revalidate cycles.

Dynamic context, batching and expert routing use declared capacity bounds for
storage. Mean expert loads can guide performance but cannot establish safety.
State schemas include reset/ownership and invocation lifetimes. Concurrent
invocations need disjoint arenas or explicit ordering; one arena cannot silently
be shared by in-flight requests.

Collectives are planned at group scope with consistent ordering, counts and types,
including ranks with no local expert work. Add communication overlap only after
independent device-buffer execution and event semantics are qualified. Graph
capture requires stable addresses and qualified mutable-input/state contracts.

## Device-buffer contract (milestone 2 contract implemented)

The implemented `inferfabric-plan::device_buffer` module provides process-independent allocation/view IDs, byte ranges, dtype, dimensions,
strides, access, placement and an explicit disjoint-output alias policy. A checked binding step
resolves these IDs into the existing POD kernel ABI without taking ownership.
Views must pass overflow, range, alignment, rank/device, access and alias checks.
Preparation may create metadata, but enqueue must not allocate, compile, transfer
through hidden host staging, synchronize or change streams. Completion governs
buffer/workspace/state lifetime, including enqueue errors.

Keep the existing kernel ABI; do not introduce a model-specific loader or a second
foreign-kernel protocol. The current kernel ABI accepts I8/BF16/F32, rank 1–8, positive contiguous strides,
and disjoint outputs/workspace. Strided and in-place variants are rejected rather
than advertised ahead of native support. Read/read overlap is allowed. Device
resource IDs resolve through borrowed caller-owned resident allocations; pointer
provenance and asynchronous completion remain the backend caller's responsibility.
Binding exports an exact view span to the existing POD ABI, not the entire arena
capacity, so disjoint subviews can safely share an allocation.

`Invocation::verify_prepared` checks the selected binary identity, numerical
contract and actual workspace requirements before enqueue; exceeding bounds fails
instead of allocating more memory. The binder independently checks actual address
ranges to catch different allocation IDs bound to overlapping storage. Serialized
contracts contain no pointers. The binding ABI remains unchanged by the resident
serial executor migration.
The viewer exposes these support limits as metadata, without claiming that opaque
backend calls have already been lowered to bound device-buffer invocations.

The complete native decoder now uses device values and resident kernel calls.
It still dispatches the token program directly rather than consuming serialized
`device_buffer::Invocation` records. Building the frozen physical runner and
preparing all resources before execution remain separate work.

## Viewer

Preserve the metadata-derived hierarchy. Add per-rank physical inspection to the
same scope: selected kernel and alternatives, costs and provenance, serial/event
ordering, memory ownership/lifetimes/offsets and unresolved requirements. Clicking
an operation connects its logical definition to physical decisions. Memory reuse
views must expose owners and ordering proofs once reuse exists. Future measured
trace overlays must match the exact plan identity; predictions are labelled.

The first viewer presents ordinal call order, not a fabricated timed Gantt chart.
It shows zero intermediate host payload separately from unknown total HBM. The physical
view paginates to 24 calls, preserving large-model usability.

## Delivery sequence and acceptance gates

1. **Physical IR and explanation:** conservative native lowering, generic contracts,
   verifier, explicit unknowns and viewer. Local deterministic and negative tests;
   validate physical metadata belongs to the displayed logical graph.
2. **Device-buffer contract and resident execution:** checked views/binding and
   device-only serial execution implemented; qualify full 92B output and transfer
   bounds. Frozen prepared-invocation execution remains a separate follow-up.
3. **Activation/workspace reuse:** fixed arenas, reuse proofs and dedicated-versus-
   pooled state/output equivalence, including asynchronous negative tests.
4. **Hardware cost database:** explicit probes, provenance and held-out prediction
   accuracy; no benchmark result manufactured from an analytical estimate.
5. **Cost-guided scheduling:** bounded search and qualified overlap; compare predicted
   and observed full-device traces, memory peaks and numerical output.
6. **Capture and serving variants:** capture-stable contracts, guarded shape variants,
   Rust frontend integration, concurrent requests and state-reset qualification.

Each milestone is committed/pushed independently. No NPU numerical requalification
is implied by local schema/viewer tests. Native numerical changes require full
model qualification and a broader corpus/context suite than the three-prompt baseline.
