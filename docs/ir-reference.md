# InferFabric IR reference

Native physical extension (2026-10-16): `inferfabric-plan::physical::Plan`, named
format `inferfabric.physical.v1`, is a verified inspection IR with generic candidate,
cost, value, boundary, collective, schedule-template and rank-binding records.
It does not replace the executable CPU v1 schema or `IFTOKN01`. See
[native physical planning](native-physical-planning.md) for current limitations and
the transition to executable device-buffer plans. The token viewer consumes this
metadata directly and does not infer hardware costs or physical allocations.

`inferfabric-plan::device_buffer::Invocation` (`inferfabric.device-buffer.v1`)
describes allocation IDs, rank/device, dtype, shape/strides, views/access,
workspace and pinned implementation identity without process pointers. Checked
binding produces the existing `IFKernelTensorV1` POD layout. The initial ABI
supports contiguous I8/BF16/F32 tensors and disjoint outputs/workspace; this
contract is available for the future physical executor, not integrated into the
current resident token runtime, which binds device resources directly in C++.

Device-residency policy `serial-device-resident-v1` rejects host/opaque kernel
candidates and unclassified activation transfers before cost search.
`inferfabric.transfer-audit.v1` records executor-owned control, parameter-preparation
and request-output transfer counts/bytes per rank. It is runtime evidence, not an
executable plan or a trace of vendor-internal metadata traffic.

Updated 2026-09-08. This catalog separates implemented formats from the proposed DSL v1 frontend and future native physical extensions. There is currently no single representation shared by every historical path. The field definitions in the linked Rust sources are authoritative; new physical-path JSON objects reject unknown fields.

## Pipeline and version ownership

| Representation | Owner / version | Status and consumer |
|---|---|---|
| Lossless CST / typed syntax AST | Proposed `inferfabric-syntax`; syntax JSON version separate from language version | Specified in the language design; parser/formatter not implemented |
| Legacy `Spec` | `inferfabric-model` | Existing line-oriented parser; mock and checkpoint-bound compiler inputs |
| Logical rank `Artifact` | `inferfabric-ir::FORMAT_VERSION = 1` | Existing structural/mock rank programs; not numerical kernel code |
| Checkpoint mathematical `bound::Plan` | `inferfabric-compiler::bound`, named format and math contract | Existing typed Qwen graph, pre-link and non-executable |
| Qwen lowered `lower::Program` | `inferfabric-compiler::lower` | Existing fixed native block runner; startup still prepares/captures operations |
| Logical graph interchange | `inferfabric-plan::ir::LogicalGraph`, version 1 | Implemented CPU planning entry point; manually authored JSON pending frontend integration |
| Typed graph | `inferfabric-plan::ir::TypedGraph`, version 1 | Resolved value IDs, inferred concrete shapes, validated update bindings |
| Optimized graph | Same `TypedGraph` schema | Dead nodes/unused constants removed, bounded constant folding, compact value IDs |
| Physical execution DAG | `inferfabric-plan::ir::PhysicalPlan`, version 1 | Frozen CPU kernels, storage, schedule and dependency metadata |
| Executable envelope / manifest | `IFPLAN01`, built-in CPU f32 ABI v1 | Checksummed binary loaded without logical graph or planner |
| Bound execution instance | `inferfabric-plan::Executor` / native future loader | Process-local allocations/state/handles; never serialized as pointer values |

The `.inferfabric` source version, semantic IR version, physical-plan version, executable envelope and kernel ABI must evolve independently. Unsupported versions fail rather than being guessed or migrated during execution.

## 1. Proposed syntax representations

The [language design](dsl-language-design.md) specifies a lossless CST containing tokens, whitespace/comments, raw foreign source, byte spans and error nodes. Typed AST accessors represent imports, model/module/kernel/implementation/plan declarations, ports, configuration, bounded composition, weights, graph calls, state policies and explicit arguments. Source maps retain definition and instantiation locations. These are syntax representations, not an execution plan. There are no CST/AST dump files until the parser is implemented.

## 2. Existing legacy model and rank IR

Sources: [model](../crates/inferfabric-model/src/lib.rs), [rank IR](../crates/inferfabric-ir/src/lib.rs).

- `Spec`: model metadata, prefill/decode meshes, page size, token budget, batch/context buckets. `Model` carries name, dimensions and the `Full`/`Delta` layer schedule. `Mesh` carries TP, CP and SP.
- `Target`: backend (`Mock`/`Ascend`), SoC, toolchain, kernel revision and flags.
- `Artifact`: version, cache key, target, spec and rank plans.
- `RankPlan`: rank identity (`Role`, TP index, CP index), state layouts and ordered instructions.
- `StateLayout`: layer, state kind (`Kv`, `Recurrent`, `Conv`), shape and dtype. KV shape is per page; recurrent/conv shape is per request.
- `Instruction`: `Kernel { layer, name }`, `Communication { group, sequence, op, tensor }`, or `Fence`.
- `CommOp`: `AllReduce`, `AllGather`, `ReduceScatter`, `Send { peer }`, `Receive { peer }`.
- `Bucket`: batch and context.

The original `PANGU` binary marker remains for compatibility after the project rename. These artifacts contain structural instructions, not the new physical CPU program. `compile`/`inspect` keep their existing meaning; `plan`/`explain`/`execute` use the separate `IFPLAN01` format.

## 3. Existing checkpoint mathematical IR

Source: [bound.rs](../crates/inferfabric-compiler/src/bound.rs).

`Dim` is `Tokens`, `Requests`, `RequestBoundaries`, `Pages`, or `Fixed(n)`. `Scalar` is `Bf16`, `Fp32`, or `I64`. `Tensor` carries name, dimensions and scalar type. `State` adds kind (`PagedKeys`, `PagedValues`, `DeltaKeyValue`, `ConvPreviousInputs`) and zero-initialization policy.

`Node` carries named inputs, typed output tensors and one operation:

| Operation | Attributes / semantics |
|---|---|
| `Embedding` | Weight binding |
| `Linear` | Weight binding; BF16 `x @ W^T`, FP32 accumulation |
| `ZeroCenteredRms` | Weight, epsilon; one-plus weight convention |
| `Reshape` | Output tensor shape defines result |
| `Split` | Axis and lengths |
| `TextRope` | Rotary dimension, theta, three interleaved sections |
| `PagedAttention` | Keys/values resources, scale, causal flag |
| `Sigmoid`, `Silu`, `Multiply`, `Add` | Elementwise mathematical contracts |
| `CausalConvSilu` | Weight, history, kernel width |
| `L2Norm` | Epsilon and reference BF16 rounding behavior |
| `DeltaDecay` | A_log, dt_bias, clamp policy |
| `DeltaRecurrence` | State, Q scale and state-layout convention |
| `GatedRms` | Weight, epsilon, normalize-before-gate policy |
| `SelectFrontier` | Last consumed token per nonempty request segment |

`bound::Plan` contains format/executable flags, math contract, reference-source hash, key, spec, checkpoint metadata, inputs, states, nodes, output, weight/state/KV byte summaries and link requirements. Checkpoint metadata describes files, tensor offsets/shapes/dtypes and bindings; it is not embedded tensor payload. This remains Qwen-specific mathematical IR.

## 4. Existing native Qwen block IR

Source: [lower.rs](../crates/inferfabric-compiler/src/lower.rs).

`Program` contains key, source key, ordered blocks, embedding binding, final normalization binding, batch, context and page size. `Block` contains layer index, `Layer` kind and weight bindings. `Binding` contains a checkpoint name and `Transform` (`Identity` or `OnePlusFp32`). This lowering requires canonical mathematical equivalence. It does not yet serialize the full native operation/memory DAG; the native runner constructs that during startup.

The existing native activation planner also has private `Interval { bytes, first, last }` and `Plan { bytes, offsets }` structures in [activation.rs](../crates/inferfabric-native/src/activation.rs). They plan hidden-state storage; they are not serialized execution programs. C++ prepared `Linear` objects retain ACL tensor/executor descriptors, workspace, child operations, device copies and HCCL gathers. Those objects and captured graph handles are process-local runtime state, not portable IR or binary sections.

The existing native activation planner also has private `Interval { bytes, first, last }` and `Plan { bytes, offsets }` structures in [activation.rs](../crates/inferfabric-native/src/activation.rs). They plan hidden-state storage; they are not serialized execution programs. C++ prepared `Linear` objects retain ACL tensor/executor descriptors, workspace, child operations, device copies and HCCL gathers. Those objects and captured graph handles are process-local runtime state, not portable IR or binary sections.

## 5. New logical and typed graph IRs

Source: [physical-path schemas](../crates/inferfabric-plan/src/ir.rs). Example: [logical graph](../examples/physical-plan/residual-state.logical.json).

All tensors in `cpu-f32-v1` are contiguous row-major f32. Shape is an array of positive static element dimensions. IDs are zero-based indexes into the representation's own value/buffer array; they may change after optimization. Names preserve diagnostic identity.

| `LogicalGraph` field | Meaning |
|---|---|
| `version`, `name` | Schema version and display name |
| `inputs` | `Tensor { name, shape }`; caller-supplied arrays |
| `constants` | `Constant { name, shape, data }`; immutable initial arrays |
| `states` | Same initializer shape; owned persistent invocation state |
| `nodes` | `LogicalNode { name, op, inputs }`; node name identifies its single output |
| `outputs` | Result value names |
| `updates` | State name to source value name; simultaneous end-of-invocation assignment |

`Op` is `add`, `mul`, `relu`, or `matmul`. This CPU primitive subset does not implement general module expansion or foreign-operation IDs. The v1 language's extension mechanism must not be reduced to this enum.

`TypedGraph` contains version/name, `values`, topologically ordered `nodes`, result IDs and state-target/source ID bindings. `Value` contains name, shape, storage (`input`, `constant`, `state`, `arena`) and an optional flattened initializer. `TypedNode` contains name, operation, input IDs and output ID. Concrete shape inference is complete at this point. `arena` here means an intermediate requiring allocation; no offset has been assigned yet.

The optimized graph uses the same schema. Folding converts intermediate values to constants and removes their calls; dead-node elimination retains output and state-update roots. Input/state interfaces survive optimization. Unused constants are removed, and IDs are compacted. The `Compilation` result retains logical, typed, optimized and physical representations for dumping; the executable payload retains only the physical representation.

## 6. Physical DAG IR

`PhysicalPlan` fields:

| Field | Contract |
|---|---|
| `version`, `name`, `target`, `policy` | Schema/display identity; currently `cpu-f32-v1` and `deterministic-cpu-v1` |
| `alignment`, `arena_bytes` | 64-byte slot alignment and total arena allocation |
| `buffers` | `Buffer { value, offset, lifetime }` |
| `steps` | Ordered launch schedule of `Step` records |
| `outputs`, `updates` | Result IDs and final state assignments |
| `removed_nodes`, `folded_nodes` | Optimization explanation names |

Arena offsets are **bytes**, shapes are **elements**, and lifetimes are inclusive **step indexes**. `lifetime = [producer,last_consumer]`; index `steps.len()` denotes final output collection/state commit. Input, constant and state buffers have null offset/lifetime, because they own separate storage. No serialized address is absolute.

`Step` contains name, kernel, input IDs, output ID, rank, stream, producer `dependencies`, `reuse_dependencies`, and selection reason. Dependencies reference prior step indexes. If two buffers share any bytes, their lifetimes must be disjoint and the later producer must depend on the earlier last consumer through a reuse edge. Merely serializing an offset is insufficient. Rank/stream are currently zero; array order supplies same-stream launch order. No multistream overlap or collectives are claimed in this CPU format.

The executable kernel catalog is `add_f32_v1`, `mul_f32_v1`, `relu_f32_v1`, `matmul_f32_v1`, and `matmul_blocked_f32_v1`. The latter uses 16x16 output tiles with tail handling. Dtypes, layouts and reduction order are part of this catalog version. The loader recomputes and verifies shape/lifetime obligations; it does not run the planner or replace these kernel choices.

Final state commit is an explicit invocation-boundary operation represented by `updates`, not a freely scheduled in-place graph write. It depends on completion of all steps. Outputs observe pre-commit state when directly returning a state buffer. New state sources are gathered before any assignment, so swaps work. Unknown native effects/events/collectives are not accepted by this schema; those require a new qualified physical extension.

## 7. Binary envelope and bound execution instance

Source: [binary.rs](../crates/inferfabric-plan/src/binary.rs).

| Byte range | Content |
|---|---|
| 0..8 | ASCII `IFPLAN01` |
| 8..16 | Little-endian u64 payload length |
| 16..48 | SHA-256 of payload |
| 48..end | UTF-8 serialized `PhysicalPlan` JSON, maximum 64 MiB |

Exact length, checksum, schema and semantic verification are required. Trailing bytes, unsupported kernels/targets, missing producers/dependencies, out-of-range views, overlapping live storage, forged lifetimes and invalid state bindings are rejected. Checksums do not establish trust. The payload is a binary execution-plan container, not an ELF and not an ACL graph-handle dump.

`04-bundle-manifest.json` is a derived inspection document: format, target, payload length/digest, built-in kernel ABI, source/planner independence, runtime binding categories, state-commit rule, empty device-object list and `device_qualified=false`. The CPU kernel functions are compiled into the Rust executor; no compiler, Python engine or dynamic kernel source is needed to execute this bundle.

`Executor` owns a decoded verified plan, arena and initialized constant/state arrays. An invocation binds a map of input names to flattened arrays and returns `RunReport { outputs, state, executed_steps, arena_bytes }`. State persists within the executor and is reset by constructing a new instance from the binary. Reports are execution evidence, not additional compiler IR.

## 8. Reference dumps and future native IR

Run `plan ... --dump-ir DIR` to capture every implemented stage of the new path. A checked-in complete example is under [physical-plan validation](validation/2026-09-08-physical-plan/ir/00-logical.json), through `04-bundle-manifest.json`. The corresponding binary, execution results and HTML visualization are in that directory's parent. Dumps contain actual produced values, not illustrative schemas.

Native physical IR must add implementation bundles, target/toolchain locks, specialization guards, weight relocations, typed views/layouts, workspace bounds, rank/stream placement, event/copy/collective instructions, state effects and graph boundaries. These are specified in [physical planning](physical-planning.md) and remain unimplemented in the new CPU executor. Extending that executor must use explicit versions and qualification rather than treating a CPU mock or a parsed declaration as native readiness.

## 9. Full-model inspection projection

`inferfabric_compiler::model_inspect::Inspection { stage, plan, groups }` is an inspection projection, not an executable IR. `plan` preserves the entire checkpoint-bound `bound::Plan`; each `Group { name, kind, nodes }` references original node indices exactly once. The canonical Qwen input-normalization bindings define layer boundaries, followed by the final normalization/head group. No operation is fused, removed or synthesized. SVG tensor edges and producer/consumer indices are derived from SSA operands; weight and state references remain explicit selectable bindings. The projection is JSON embedded in a self-contained HTML artifact.

[Qwen3.5-2B's complete typed IR and viewer](validation/2026-09-08-qwen-plan/README.md) provide a full-model reference in addition to the small CPU physical-plan dumps. A typed tensor shape is not an allocation or a measured peak; native physical details remain a separate pending IR export.
## Exact prefill family binary extension

`IFTOKN06` preserves the version-4 instruction encoding and extends its header.
After the eight-byte magic, all integers are little-endian `u32`: context, EOS,
output instruction, instruction count, resident count, maximum prefill width,
shared-page flag, variant count, then the exact widths. The instruction stream
follows immediately. The flag is 0 or 1. There are 1–4 strictly increasing widths
chosen from 2, 4, 8, 16, and the last must equal maximum prefill width. Decode's
single-token entry is separate. Readers reject malformed lists before loading
weights; versions 2–5 retain their existing bytes and exact-size semantics.

`device-program.json` records `prefill_variants`; `serving.json` carries a separate
exact invocation and numerical-class identity per width. An empty/absent family
field means the older single-size contract. The native ABI returns a bit mask
with bit `n` set only for declared width `n`; Rust checks this against the manifest.
Each instruction recipe exposes `prefill_variants`, including invocation shapes,
API repetition and assembly-copy counts. Persistent state/control shapes and
offsets stay sized for the family maximum. Every graph, retained output and output
invocation participates in workspace discovery and frozen-arena preparation before
readiness. The executor rejects missing graphs after preparation is sealed.

## Completed arithmetic history

`inferfabric_ir::arithmetic_history::ArithmeticHistory` records completed input
positions as canonical intervals `{start, end, query_tokens, numerical_class}`,
with `version: 1`. Adjacent identical shapes/classes coalesce. This preserves
invocation boundaries without one record per token: 4,096 consecutive T=16
invocations cover 64K in one interval. Prefix extraction rejects a boundary
inside a packed invocation, a boundary beyond completed work, and incompatible
interval histories even when the last numerical class matches.

The Rust native adapter tracks one history per resident, preflights contiguity
before dispatch, appends only after every rank's owning stream completed, and
clears history on successful state reset. Its accessor exposes completed metadata;
it does not grant device checkpoint readiness or cross-class numerical equivalence.
Future prefix lookup must compare this history together with `PrefixIdentity`,
exact tokens, state completeness, alignment, ownership and completion epoch.

`INFERFABRIC_ARITHMETIC_AUDIT_DIR` optionally writes
`inferfabric.completed-arithmetic.v1` records when resident histories are released.
Records contain resident/sequence IDs and arithmetic intervals, never prompt token
IDs or tensor payloads. Exclusive file creation refuses to overwrite older runs.
The HTTP qualifier's `--check-arithmetic-history` verifies declared shapes/classes,
contiguous complete intervals, unique records and equality between their token sum
and the scheduler's consumed-token count, including cancelled requests.

`inferfabric_runtime::state_pool::SealedLease` is an opaque immutable allocation
identity. `seal` rejects incomplete rank work; `submit` rejects writes to sealed
state, and `borrow_sealed` retains read-only ownership after the original owner
retires. There is no unseal operation. Reclamation requires all readers and rank
completions; a later allocation has a fresh generation even when page IDs repeat.
The `sealed_bytes` usage counter is a subset of occupied bytes. This token proves
ownership/completion constraints, not that the device holds a complete model
checkpoint. Native snapshot and restore operations remain a separate contract.

## Compact request-output rows: token program v8

`IFTOKN08` extends the uncaptured-prefill v7 header. After the shared-page flag,
four little-endian u32 values encode `source`, `source_width`, `first`, and `last`.
The ordinary instruction stream follows unchanged. JSON `output_rows` additionally
names the row domain (`request_sample_rows`), selection implementation
(`ordered_device_row_copy`) and empty-row behavior. The header is part of the
execution digest; older readers must reject it rather than ignoring the contract.

The compiler proves a contiguous unary output suffix ending at the program output.
The source remains a full-token tensor. For uncaptured execution, ordered sample
indices select device rows into compact storage before the suffix; its operations
use the compact row count. With zero requested rows the full prefix executes and
only this suffix is skipped. Returned logits use compact member order, not original
packed offsets. Fixed-count decode graphs retain their normal row mapping.

Unknown/stateful operations and compound quantized values are boundaries. A
rank-major `gather_heads` and its matching `head_to_token` conversion move together;
selection may not split the layout pair. Neither runtime nor optimizer may drop
state updates or replace reduction precision. Native loading checks the range,
source shape, unary dependencies and approved operation/layout sequence.

Preparation must cover all sampled-row counts through resident capacity before
binding memory. The viewer exposes selection and dynamic row axes in each affected
recipe. This candidate is enabled at compilation by
`INFERFABRIC_EXPERIMENTAL_OUTPUT_ROWS=1`; NPU qualification, not successful
compilation alone, determines whether a generated bundle is accepted for serving.
