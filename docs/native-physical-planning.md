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

## Current milestone 1

`inferfabric-plan::physical` supplies versioned, backend-neutral inspection IR,
a deterministic legal candidate selector and a serial schedule verifier. The
Ascend token adapter registers the current operator implementations without model
names or layer schedules. `compile-token` publishes `physical.json` alongside
`execution.html`; the viewer embeds the same verified physical metadata.

The existing 92B diagnostic path uses host vectors, internal device allocations,
transfers and frequent synchronization. Consequently this first physical artifact
is explicitly **inspection-only and non-executable**. One physical step represents
a backend call with opaque internals, not a claim that each logical op is one
NPU kernel. Boundary annotations identify host/device transfers, synchronization,
allocation and persistent-state handling. They do not pretend to be a complete
internal launch DAG. The native runtime continues consuming `execution.bin`.

Actual decisions now recorded:

- Selected implementation, contract/source fingerprint, numerical contract,
  legal alternatives, rejection reasons and selection rationale.
- Explicit rank/device bindings to shared immutable schedule templates. Identical
  schedules are stored once; the rank selector does not imply measured rank times.
- Data dependencies, serial-order dependencies, stream zero and collective
  group/signature/sequence. Current lowering preserves all-rank collective order.
- Known host output payload shapes/dtypes, inclusive last-use positions, retained
  outputs and partial peak host payload. Opaque handles, container overhead,
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

## Device-buffer contract (milestone 2 scope)

Provide process-independent allocation/view IDs, byte ranges, dtype, dimensions,
strides, access, placement and explicit alias permissions. A checked binding step
resolves these IDs into the existing POD kernel ABI without taking ownership.
Views must pass overflow, range, alignment, rank/device, access and alias checks.
Preparation may create metadata, but enqueue must not allocate, compile, transfer
through hidden host staging, synchronize or change streams. Completion governs
buffer/workspace/state lifetime, including enqueue errors.

Keep the existing kernel ABI; do not introduce a model-specific loader or a second
foreign-kernel protocol. Contract implementation and local binding tests do not
claim that the complete native decoder already uses device buffers. Replacing
host-staged kernels and building the executable physical runner follows separately.

## Viewer

Preserve the metadata-derived hierarchy. Add per-rank physical inspection to the
same scope: selected kernel and alternatives, costs and provenance, serial/event
ordering, memory ownership/lifetimes/offsets and unresolved requirements. Clicking
an operation connects its logical definition to physical decisions. Memory reuse
views must expose owners and ordering proofs once reuse exists. Future measured
trace overlays must match the exact plan identity; predictions are labelled.

The first viewer presents ordinal call order, not a fabricated timed Gantt chart.
It shows partial host payload accounting separately from unknown HBM. The physical
view paginates to 24 calls, preserving large-model usability.

## Delivery sequence and acceptance gates

1. **Physical IR and explanation:** conservative native lowering, generic contracts,
   verifier, explicit unknowns and viewer. Local deterministic and negative tests;
   validate physical metadata belongs to the displayed logical graph.
2. **Device-buffer contract, then execution:** generic checked views/binding first;
   separately migrate kernels and serial execution, preserving full 92B output.
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
