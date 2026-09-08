# Physical planning and execution

Updated 2026-09-08. The SQL-style plan/execute boundary is part of the target architecture. A working local implementation exists in `inferfabric-plan`: static f32 CPU graphs, checked physical planning, binary loading/execution, and standalone HTML explanation. It does not replace the existing Ascend/Qwen serving path or implement the proposed v1 text parser.

## 1. The boundary

A DSL `plan` declaration is a request containing target, parallelism, shape bounds, and execution policy. The compiler must derive a physical execution plan from it. The declaration itself is not an execution DAG.

```text
Source / syntax tree
  -> elaborated logical graph
  -> typed and effect-checked graph
  -> optimized logical graph
  -> target-specialized physical DAG and launch schedule
  -> verified executable bundle
  -> load / bind / prepare
  -> execute invocation(s)
```

Planning decides implementations, fusion, layouts, sharding, communication, dependencies, launch placement, and storage. Binary emission serializes those decisions after verification. Execution consumes them; it must not reconstruct a model from its display name, rerun optimization, change a layout, or choose another kernel because one is unavailable.

An inference invocation is a bounded step or prefill chunk. Generation is not statically unrolled for an unknown output length. The scheduler binds tokens, positions, valid lanes and request state to an eligible planned variant on each invocation.

## 2. Physical planning passes

The target planner has explicit passes and records their decisions:

1. **Bind and specialize.** Validate checkpoint mappings; substitute model constants and bounded serving capacities. Keep tokens, lengths and request state as dynamic operands.
2. **Simplify.** Eliminate unreachable pure computations, fold safe constant expressions, remove redundant conversions and preserve state/effect roots.
3. **Choose physical implementations.** Match qualified kernel contracts, enumerate legal layouts and fusion patterns, and select deterministic baseline implementations. Later cost models may compare latency, launch overhead, communication and memory, with provenance for measurements.
4. **Partition.** Assign ownership and tensor layouts to ranks. Insert explicit copies, layout conversions and collectives with rank-consistent ordering. Reject unsupported partition rules.
5. **Schedule.** Derive a DAG from tensor dependencies and effects, assign streams, and insert event waits/records. A serial schedule is a valid baseline, not evidence of overlap optimization.
6. **Plan storage.** Separate weights/constants, persistent request state, activation arenas, communication buffers and workspace. Allocate aligned offsets using inclusive lifetimes. Add reuse dependencies whenever one value reclaims another's bytes.
7. **Reconcile resource requirements.** Kernel selection, fusion, overlap and workspace requirements interact. Revisit choices if they exceed memory constraints. Finish with concrete sizes or verified bounded resource contracts, never unresolved allocation guesses.
8. **Verify and emit.** Check initialization, bounds, alias legality, effects, collective agreement, specialization guards, kernel availability and output/state coverage. Freeze the physical plan and bundle identities.

Future planners may search alternatives; the first implementation uses deterministic rules. `explain` must distinguish static rules, measured costs and unavailable estimates. No estimated latency is presented as a measurement.

## 3. What the executable bundle contains

The target bundle contains per-rank physical DAGs, selected kernel IDs/native objects, launch metadata, tensor views/layouts, arena offsets, copy/collective/event instructions, state schemas, invocation boundaries, shape guards, weight relocations, workspace bounds, target/toolchain locks and qualification constraints. Source locations and optimization reasons can remain as debug sections.

Raw device pointers, communicator handles, ACL executors and captured graph handles are process-local resources. The loader binds actual allocations and weights, establishes communicators, prepares validated launch descriptors, and recreates captures. Preparation cannot silently choose a different execution plan. If a vendor reports workspace larger than the recorded bound, loading fails and an external planning stage must produce another bundle. A recorded graph boundary is portable metadata; a captured driver handle is not.

Checksum validation detects corruption, not authorship. Native bundles also need dependency/ABI validation and the deployment's trust policy. The current CPU bundle never loads arbitrary native objects; it references a fixed versioned kernel catalog in the executor.

## 4. Locally implemented path

The `plan` command currently accepts the documented logical graph JSON interchange, not `.inferfabric` v1 source. A future parser/elaborator will produce this kind of typed graph through the generalized contract IR. The closed CPU primitive enum is a bootstrap executor catalog, not the final extensibility mechanism for foreign operations.

Supported operations are contiguous f32 `add`, `mul`, `relu`, and `[M,K] @ [K,N]` matmul. Elementwise operands require identical shapes; there is no implicit broadcasting. Declared inputs and persistent state retain their interfaces. Nodes may be listed in any order; names are resolved and a deterministic topological order is chosen. Duplicate value names, cycles, unknown operands and invalid shapes fail.

The planner removes dead pure nodes, folds all-constant operations of at most 4096 output-element/MAC work units, and discards unused constants after folding. Matmul with at least 4096 MACs selects the 16x16 output-tiled implementation; smaller matmul uses the scalar-loop implementation. Both preserve increasing reduction order. This threshold is a transparent static policy, not an autotuned performance result. No fusion or distributed selection is implemented in this CPU path.

Physical steps freeze kernel IDs, buffer IDs, rank 0, stream 0, producer dependencies, memory-reuse dependencies and selection reasons. Array order is the launch schedule; the single stream adds serial order between launches. DAG edges show data and storage dependencies without implying that this executor runs branches concurrently. The allocator assigns 64-byte-aligned offsets, keeps operands live through their consumer, and retains outputs/update sources until the invocation boundary. It does not perform in-place kernel aliasing.

The executor allocates the arena and state once. Kernel loops read bound buffers and write planned offsets directly, without per-operation tensor allocation or replanning. Input/output JSON and report collection are CPU control-path allocations. State assignments are simultaneous at invocation end; all sources are collected before any state changes. Invalid inputs or nonfinite intermediate results return an error without committing state. This stronger local CPU commit behavior is not a promise of rollback after partial NPU writes.

Current limits: 512 operations, 2048 values, tensor rank 1..8, positive static dimensions, 64 MiB logical tensor storage and arena limits, 100 million MACs per matmul, and a 64 MiB serialized payload. Unsupported versions, targets, ranks, streams and kernel IDs fail before execution. These limits define the local verification target, not an LLM capacity claim.

## 5. Commands and complete IR dumps

From a local Rust environment:

```sh
cargo run -p inferfabric-cli -- plan \
  examples/physical-plan/residual-state.logical.json \
  .deploy/physical-plan/demo.ifplan --dump-ir .deploy/physical-plan/ir
cargo run -p inferfabric-cli -- explain \
  .deploy/physical-plan/demo.ifplan --html .deploy/physical-plan/plan.html
cargo run -p inferfabric-cli -- execute \
  .deploy/physical-plan/demo.ifplan examples/physical-plan/inputs.json \
  .deploy/physical-plan/results.json
```

`execute` takes an array of invocation input maps; each map binds names to flattened f32 arrays. State persists between maps in that process. The source JSON may be removed after planning. The integration test does exactly that before invoking `explain` and `execute` in separate processes.

`--dump-ir` emits `00-logical.json`, `01-typed.json`, `02-optimized.json`, `03-physical.json`, and `04-bundle-manifest.json`. [The IR reference](ir-reference.md) documents every representation, including existing legacy and Qwen IRs. The HTML view is self-contained, requires no external scripts, and supports node inspection, search, fit/zoom, memory-edge toggling, buffer/lifetime inspection and JSON download. It visualizes the decoded binary, not a reconstruction from source.

The example computes `relu(x @ weights + bias) + x * (scale_a + scale_b) + history`, then commits the result to history. One constant node is folded and one dead node removed. Six executable steps use a 192-byte arena versus 384 bytes of individually aligned temporaries. Two identical inputs yield `[10,13,2,2]` and `[20,26,4,4]`. This is actual CPU arithmetic, distinct from the repository's synthetic mock scheduler.

## 6. JIT and native integration

JIT is a future producer of physical-plan variants through these same passes. Compilation and tuning stay outside enqueue. A variant is verified, linked, warmed, qualified and captured before atomic publication. The executor accepts only eligible published variants. Cache identities include semantic graph, specialization, layouts, implementation dependencies, compiler/toolchain/target identities and qualification constraints. Retire code, graphs and storage only after their last invocation completes.

Native follow-up work is explicit: adapt the general typed/effect IR; add qualified native kernel/collective/event descriptors; derive workspace bounds; emit per-rank plans; integrate loader relocation and capture; and compare intermediate tensors/state against the existing Qwen path. The CPU implementation is not evidence that the new binary format executes Qwen on Ascend. No NPU execution or remote deployment was performed for this milestone.

## 7. Inspecting a complete planned model

`explain-model TYPED-PLAN.json --html MODEL.html` renders the existing checkpoint-bound Qwen plan at model scale. A layer overview drills into exact typed operator DAGs; input boundaries preserve producer links across layers. Node details expose all attributes, symbolic tensor shapes, checkpoint weight descriptors and persistent state references. Global search and producer/consumer navigation include tied weights and state bindings. The complete typed plan is embedded and downloadable. See the [Qwen3.5-2B reference](validation/2026-09-08-qwen-plan/README.md).

This complements `explain PLAN.ifplan`, which inspects the new executable CPU physical-plan format. It does not route Qwen through that CPU executor. The model inspector groups canonical operations by explicit layer-entry normalization bindings, validates graph equivalence, and displays logical memory formulas. Future native physical-plan export should enrich the same model navigation with selected kernels, fusion mapping, per-rank placement, arena lifetimes/offsets, workspace bounds and graph-capture regions. Those details must come from the actual planner/runtime descriptors.
