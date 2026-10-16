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

The resident 92B path now uses reusable per-rank device buffers, cached qualified
invocations and same-stream ordering, with completion only at request output.
See [prepared device execution](prepared-device-execution.md) for ownership and
qualification details. The physical artifact remains **inspection-only and non-executable**
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

## Measured optimization priorities (2026-10-16)

The full resident 92B TP8/EP8 decode profile measures approximately 2.06 s per
step, dominated by collective submission, per-call synchronization and temporary
allocation/copies. The [measured optimization plan](92b-decode-optimization-plan.md)
orders prepared buffer execution, capture, distribution-aware collective removal,
device-routed expert dispatch and MLA fusion, with numerical and residency gates.
Its [evidence](validation/2026-10-16-92b-profile/README.md) distinguishes lightweight
wall timings from intrusive CANN device tracing. These measurements guide upcoming
physical planning work; they do not certify an optimized prepared executor.

## Executable storage lowering and inspection — 2026-10-16

`execution.bin` now uses `IFTOKN02`. The compiler produces a separate
`device-program.json` with executable instruction IDs, original logical-node
IDs, selected implementation IDs, storage dtypes, dependencies and reasons.
`execution.json` retains the authored logical graph. Initialization remains
checkpoint-driven and unchanged. The browser consumes these artifacts; it does
not reconstruct architecture or infer sharding from model names.

The executor consumes explicit cast/copy instructions. Projection selection is
bound to checkpoint dtype before serialization; unbound choices cannot be
encoded. Native validation independently checks input/output storage, selected
kernel identity and checkpoint compatibility before device allocation. Runtime
input/output guards catch implementation drift. Unsupported types fail closed.

Conversion CSE shares casts only for the same immutable SSA producer and target
dtype. It never cancels FP32 → BF16 → FP32 across an authored rounding boundary.
Residual addition lowers to FP32 inputs, FP32 add and an explicit BF16 round.
The existing vendor numerical contract is preserved.

For in-place convolution, planning proves exclusivity of a fresh allocation or
a bounded slice of one. All readers of the same allocation are checked for byte
range overlap; nested slices contribute offsets. Disjoint sibling views may
remain live. Output escapes, overlapping consumers and opaque/persistent roots
prevent mutation. A failed proof emits a private device copy; a successful proof
passes the owned region directly. Native validation repeats the ownership proof.
This is a real execution decision, not a viewer annotation. Device-pool leases
retain backing allocations and reuse is ordered on the owning stream.

The viewer starts with module summaries and expands to at most 25 instruction
rows per page. Kernel APIs, precision/storage, materialization, rank placement
and collectives are columns, with filters for casts, communication and copies.
Selection details include dependency navigation and checkpoint bindings. TP/EP
sizes and expert counts come from backend/graph metadata. The current EP method
is replicated routing plus owner-local computation and explicit AllReduce;
All-to-all and AGRS are explicitly unsupported. HCCL transport remains a runtime
binding. A vendor API is identified as an API, not falsely counted as one device
kernel launch.

### Remaining work recorded at the storage-boundary milestone (updated below)

This revision makes operation-boundary storage executable. It does **not** yet
make all vendor recipes primitive instructions. The following are mandatory
completion stages, not capabilities claimed by the new viewer:

1. Expand composite recipes into typed device values and invocations: query
   projection/rotary/packing, routing/quantization, grouped expert execution,
   mHC mixing and persistent cache writes. Separate preparation-only parameter
   transforms from per-token work. Every temporary must have a shape, storage
   dtype, strides, alias region, mutability and lifetime; unknown vendor
   workspace must remain explicitly unresolved.
2. Move all remaining casts/materializations into that IR. Kernel contracts
   distinguish storage precision, computation/accumulation precision and final
   rounding. Kernel fusion must preserve this numerical contract. The executor
   must reject undeclared conversions and copies, including recipe-internal
   ones; current guards cover the inter-operation boundary only.
3. Allocate activation/workspace arenas from that complete DAG. Reuse needs
   non-overlapping lifetimes and stream/event ordering; aliases share an
   allocation, persistent state never shares transient storage. Export capacity,
   offsets, live ranges and reuse dependencies for each rank. The current pool
   is runtime prepared and is not a frozen arena.
4. Select among qualified kernels/layouts and EP methods using compatible
   measured cost samples keyed by hardware/runtime/kernel revisions, shapes,
   dtypes and topology. Unknown cost cannot win as zero. Collectives carry group,
   sequence, dtype, volume and ordering; compare AllReduce, All-to-all and AGRS
   only after their implementations and correctness gates exist. Activation
   host fallback remains inadmissible before cost evaluation.
5. Bind the complete physical program and vendor identities to the executable
   bundle. Replay must issue only declared work; reject mismatched identities,
   stale plans and undeclared transfers. Verify full-model outputs, graph replay,
   residency and execution traces against the plan before calling it complete.

### Hardware-bound arena and query expansion (2026-10-16)

Dense INT8 projections now have separate quantization and matmul instructions.
MLA query construction expands into query projection, strided rotary extraction,
FP32 rotary embedding, BF16 rounding, and column concatenation. The generated
instructions retain their logical source node and module hierarchy. This exposes
the temporary values and their dependencies without changing the arithmetic.

Hardware preparation records every prepared allocation slot, including persistent
state, packed parameters, activation temporaries and vendor workspace. Slots have
512-byte-aligned, disjoint offsets. The initial policy preserves the qualified
exact-size LIFO reuse schedule; it does not yet optimize the footprint or discard
obsolete workspace-growth slots. Checkpoint storage is allocated separately by
the initialization executor. ACL/HCCL internal allocations are outside this arena.

After a full prepared audit, bind a new bundle inside the runtime container:

```sh
python3 scripts/bind-prepared-arena.py PLAN AUDIT_DIR BOUND_PLAN
```

The output contains `arena.json` and one `arena-rank-N.txt` per rank, alongside
the original executable and initialization plans. Each native process reserves
one contiguous arena, creates descriptors over its fixed offsets, and never
grows it. Allocation requests must match a free declared slot of exactly the
requested size and an instruction-to-slot binding observed during preparation.
Workspace uses the same arena. Aliases retain leases; a slot becomes reusable
only after the last lease is released on its owning stream. Graph capture retains
the arena addresses through replay; no global fence or activation host transfer
is introduced.

The binding includes fingerprints of both `execution.bin` and the native backend
binary. A changed binary/program requires fresh preparation. Overlap, alignment,
extent, duplicate bindings, unknown slots, exhausted capacity and unplanned
instruction bindings fail closed. Use the same qualified CANN/vendor/HCCL runtime
for preparation and execution. Runtime-internal allocation costs are not covered
by this layout, and this is not an offline estimate of unknown vendor workspace.

The viewer reads the bound arena metadata and exposes rank capacity, slot offsets,
sizes, and the slots acquired by each instruction during preparation. This usage
table includes preparation-only buffers; it is not a claim that every listed slot
is newly allocated on each token. Some backend recipes remain composite, so this
milestone must not be described as completion of the full primitive execution DAG.

### Per-token composite expansion and strict execution (2026-10-16)

The subsequent expansion completes the **per-token vendor-API DAG** for the
qualified decoder. It emits 2,661 instructions from 1,701 logical operations:

- mHC projection and Sinkhorn are separate instructions with an exclusive
  intermediate handle.
- Router projection and device top-k are separate, with FP32 logits between them.
- Route filtering/quantization, activation/quantization, output zeroing, grouped
  matmul, packed-row extraction, FP32 conversion and weighted expert reduction
  each have explicit inputs. Grouped matmul can mutate only its exclusively used
  zero-initialized output. Separate gate/up projections share route quantization
  and explicitly pack their rows before activation.
- Cache input packing is a separate concatenation. Attention metadata has one
  zeroing operation and one producer, shared by all attention consumers through
  explicit dependencies.
- Dense quantization, query rotary extraction, rotation, rounding, concatenation,
  residual-add rounding, views and collectives remain explicit instructions.

The runtime rejects implicit dtype conversions. Device copies are permitted only
by copy/packing opcodes or a declared parameter/state preparation scope on the
first token before capture. No lazy preparation copy is permitted during capture
or later tokens. Grouped expert scales now load and pack directly as BF16: the
previous BF16→FP32→BF16 parameter round-trip and extra FP32 bank are eliminated.
Dense INT8 matmul retains its distinct FP32 scale contract.

Normal execution requires a complete frozen arena bundle. An unbound bundle may
run only with explicit `INFERFABRIC_PREPARE_ARENA=1` hardware preparation (or the
benchmark's `--prepare-arena` switch). Missing rank files cannot silently select
the pool. Preparation measures vendor workspace; binding freezes addresses and
allowed instruction-to-slot assignments. Graph and non-graph execution use the
same contract.

This expansion stops at the declared vendor API boundary: a vendor operation may
launch multiple internal device kernels. Parameter folding/packing, sink constant
normalization and recurrent-state initialization remain explicit **preparation
recipes**, shown separately in the viewer. Their internal graph is not claimed to
be a fully lowered initialization DAG. The arena preserves the qualified reuse
policy rather than performing optimal interval packing or measured cost search.

## Planned maximum shared workspace (2026-10-17)

Hardware preparation now has a discovery stage followed by a fresh preparation
with a fingerprint-bound maximum workspace reservation. Arena binding rejects
growing/discovery pools. One permanent workspace lease serves all instructions
on the owning stream, and an oversized demand fails before kernel enqueue.
The viewer records capacity, per-instruction demand and the shared slot.
See [the protocol](shared-workspace-planning.md).
