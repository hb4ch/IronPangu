# InferFabric language v1: model graphs and extensible kernels

Status: proposed language design, 2026-09-08. The separate [physical planning path](physical-planning.md) now has a locally verified CPU implementation. No parser, kernel adapter, or device capability described here is implemented by this document. This proposal replaces the current line-oriented model description. Implementation sequence: [DSL implementation plan](dsl-implementation-plan.md).

## 1. Decision and scope

Build a small, typed, declarative graph language, with ordinary named modules and a separate execution plan. Parse source into a lossless concrete syntax tree (CST), expose a typed abstract syntax tree (AST, the requested syntax DOM), and elaborate that tree into a checked graph IR. Do not parse directly into the current `Spec`.

Three authoring levels share one language:

1. Model authors compose named layers, bind checkpoint weights, and choose an execution plan.
2. Layer authors define typed graphs, parameters, persistent state, and mathematical contracts using reusable modules.
3. Kernel authors implement a declared operation using raw Triton-Ascend, TileLang-Ascend, Ascend C, or PTO source. Implementations live behind a versioned adapter and native ABI.

A new layer must not require adding a variant to a Rust `Layer` enum. A new implementation of an existing operation must not change its mathematical contract. A new backend adapter can require compiler development; a new kernel using an installed adapter must not.

The language describes inference computations and resource contracts. It is not a general-purpose Python replacement or a kernel programming language. No implicit model selection by display name, arbitrary host evaluation, automatic broadcasting, or runtime compilation is part of v1.

## 2. Why the current representation cannot be extended safely

At the reviewed source revision, `inferfabric-dsl/src/lib.rs` splits lines on `=` and immediately builds `Spec`; comments are stripped before any string grammar exists. It has no syntax tree, reusable declarations, typed ports, or source ranges beyond a few line errors. `inferfabric-model::Layer` contains only `Full` and `Delta`.

`inferfabric-compiler/src/bound.rs` binds a particular Qwen checkpoint and emits its fixed graph. `lower.rs` recompiles that graph to establish canonical equality and lowers it into fixed native blocks. The native runner also knows those block shapes. Consequently, adding prettier input alone would neither express a novel layer nor execute it correctly.

Preserve the exact existing Qwen semantics during migration, including weight tying, one-plus normalization, Q/gate packing, partial RoPE, BF16 normalization behavior, convolution history, and FP32 recurrence. Replace the representation and extension boundary without silently changing the model mathematics.

## 3. Surface syntax and readability

Use `.inferfabric`, UTF-8, braces, semicolons, `name = value`, named arguments, and four-space indentation. Newlines have no grammatical significance. Comments use `//` or non-nested `/* ... */`; strings use JSON escapes. Identifiers are ASCII `[A-Za-z_][A-Za-z0-9_]*`; Unicode is allowed in comments and strings. Hyphenated model labels are strings, never identifiers.

Every file begins `language inferfabric 1;`. Unsupported versions fail with a migration diagnostic. Imports are explicit, aliased, and resolved from a local package lock; parsing never fetches or executes them. Unknown declarations, fields, arguments, and duplicate names are errors. No implicit last-definition-wins behavior. Formatting preserves comments and raw source bytes.

The following is a complete proposed model/plan authoring example, assuming the versioned standard modules described below. It is a specification fixture, not executable with today's parser:

```inferfabric
language inferfabric 1;
import "std/qwen35@1" as qwen;
import "std/tensor@1" as tensor;

model Qwen35_2B {
    label = "Qwen3.5-2B text";
    const H = 2048;
    axis T = runtime();
    axis R = runtime();
    input tokens: tensor<i64>[T];
    input context: RequestContext<T, R>;
    output logits: tensor<bf16>[R, 248320];

    module embedding = qwen.Embedding(vocab = 248320, hidden = H);
    stack blocks = repeat(count = 6, items = [
        qwen.DeltaBlock(hidden = H, intermediate = 6144,
                        heads = 16, head_dim = 128, conv_width = 4),
        qwen.DeltaBlock(hidden = H, intermediate = 6144,
                        heads = 16, head_dim = 128, conv_width = 4),
        qwen.DeltaBlock(hidden = H, intermediate = 6144,
                        heads = 16, head_dim = 128, conv_width = 4),
        qwen.AttentionBlock(hidden = H, intermediate = 6144,
                            query_heads = 8, kv_heads = 2, head_dim = 256),
    ]);
    module norm = qwen.FinalNorm(hidden = H, epsilon = 1e-6);
    module head = qwen.TiedHead(hidden = H, vocab = 248320);

    weights {
        source = checkpoint(format = "safetensors", config = "config.json");
        contract = qwen.TextCheckpointV1;
        mapping = qwen.HuggingFaceTextV1;
        tie = [share(left = head.weight, right = embedding.weight)];
        unexpected = "reject_except_declared_namespaces";
    }
    graph {
        let x = embedding(tokens = tokens);
        let h = run(stack = blocks, x = x, context = context);
        let n = norm(x = h);
        let last = tensor.last_token(x = n, context = context);
        yield logits = head(x = last);
    }
}

plan LocalTP2 for Qwen35_2B {
    target = target_manifest(path = "targets/ascend-local.json");
    prefill = mesh(tp = 2, cp = 1, sp = false);
    decode = mesh(tp = 2, cp = 1, sp = false);
    serving {
        page_tokens = 128;
        token_budget = 1024;
        batch_buckets = [1, 2, 4, 8];
        context_buckets = [128, 512, 2048, 8192];
    }
    execution {
        graph_capture = "required";
        missing_implementation = "error";
    }
}
```

`repeat` is bounded compile-time expansion. It creates 24 distinct instances, with stable paths `blocks.0` through `blocks.23`; it never shares parameters or state accidentally. `run` requires every instance to implement exactly the stack ports `x`, `context`, and one tensor result; module signature checking establishes compatibility between adjacent layers. General branching uses ordinary graph nodes rather than `run`. More than one output is bound by a tuple pattern in declaration order.

`qwen.*@1` denotes pinned mathematical definitions, not an opaque architecture switch. The package must ship inspectable module graphs, checkpoint mappings, state schemas, numerical rules and tests. Its attention defaults fix the existing Qwen RoPE/QK-normalization/gating semantics. Defaults are allowed only on named module configuration parameters; expanded IR records every resolved value. The package has a content digest in the lockfile, so `@1` does not mean “whatever was installed most recently.”

Runtime checkpoint location is supplied explicitly to the compiler. `config.json` is relative to that checkpoint, not an unchecked model-source filesystem path. The target manifest names exact SoC/toolchain/backend capability identifiers; `Ascend` alone does not qualify a kernel.

## 4. Reusable layer modules and typed composition

Configuration parameters are compile-time values. Inputs, outputs and parameters are tensors; state is a separate resource. Example of a custom graph-defined layer, without any new compiler enum:

```inferfabric
module GatedMLP(H: dim, I: dim) {
    axis T = runtime();
    input x: tensor<bf16>[T, H];
    param gate: tensor<bf16>[I, H];
    param up: tensor<bf16>[I, H];
    param down: tensor<bf16>[H, I];
    output y: tensor<bf16>[T, H];
    graph {
        let g = tensor.linear(x = x, weight = gate, accumulate = "fp32");
        let u = tensor.linear(x = x, weight = up, accumulate = "fp32");
        let a = tensor.silu(x = g);
        let z = tensor.multiply(left = a, right = u);
        yield y = tensor.linear(x = z, weight = down, accumulate = "fp32");
    }
}
```

This snippet belongs in a versioned file importing `std/tensor@1`. The `linear` contract specifies `[out,in]` storage and BF16 results with FP32 accumulation. A layout conversion, broadcast, dtype cast, reduction ordering change, quantization, or transpose must be explicit or a proved compiler rewrite under that contract.

Types include scalar `bool`, `int`, `dim`, `string`, `dtype`; tensor element types; `tensor<dtype>[dimensions]`; tuples; and registered resource types such as `RequestContext<T,R>` and `PagedKV<...>`. Initial tensor dtypes are BF16, FP32, I32, I64 and bool; future dtypes require versioned semantics. A `dim` is a positive checked integer; zero-token requests are handled at admission and are not zero-sized graph launches.

`axis` binds symbolic runtime dimensions, which unify across connected ports. They must acquire finite bounds from the execution plan before memory planning. Per-request lengths are not shapes: packed-token boundaries, positions, masks, and page tables are typed context operands. Do not conflate padded bucket size with valid tokens. A kernel receives both capacity and validity metadata when needed.

Dimension expressions use checked integer arithmetic. The initial constraint solver supports equality, bounds, divisibility, affine expressions and explicit constant products; unsupported proofs are reported, not guessed. `H / tp` requires a proof of exact divisibility. Physical layouts are separate from logical shapes and are explicit at external-kernel boundaries. No silent CPU copy or dtype fallback is permitted.

Graph values have single assignment and lexical scope; forward value references and graph cycles fail. Declarations can reference later declarations after symbol collection. Module recursion and import cycles fail in v1. `const` and bounded graph construction do not evaluate tensor data. Tensor-dependent control flow, recursive graphs, training/autodiff and dynamic expert routing are future versioned IR features, not arbitrary Python escape hatches.

## 5. Persistent state, ownership, and parallelism

Stateful modules declare resources and pass explicit access capabilities:

```inferfabric
module DeltaMemory(H: dim) {
    axis T = runtime();
    axis R = runtime();
    input x: tensor<bf16>[T, H];
    input context: RequestContext<T, R>;
    state recurrence: tensor<fp32>[R, H, H] {
        scope = "request";
        initialize = "zero";
        reset = "on_request_reuse";
        transfer = "required";
        schema = "example.delta-memory.v1";
    }
    output y: tensor<bf16>[T, H];
    graph {
        yield y = custom.delta(x = x, context = context,
                               memory = update(recurrence));
    }
}
```

This is an illustrative custom resource shape, not the Qwen recurrence shape. `custom.delta` must be imported and declare the matching resource port and read/write effect. Each call consumes an exclusive update capability and produces an implicit effect token in IR. Two unordered writers to the same resource are rejected. Persistent updates are committed only after successful execution completion; cancellation waits for outstanding operations before reset/reuse. Failures may invalidate a request's state; the system does not promise rollback of partially executed device writes.

Ownership attaches to model-instance path, request generation, resource schema, and shard. `read(state)` borrows immutably; `update(state)` borrows exclusively; raw tensor references cannot erase ownership. Model/module state is initialized once per request, not per token or graph replay. Context carries validity so padding never mutates an inactive request. Weight sharing is explicit and read-only; state sharing is unsupported in v1.

The state descriptor includes logical axes and ordering, dtype, physical layout, initialization, reset, serialization version and transfer policy. KV pages, convolution history and recurrence all participate in transfer. A custom state format must provide a serializer/layout mapping or reject PD deployment. Unknown state is never omitted from the manifest.

Plans request parallelism; modules and kernel contracts declare valid sharding behavior. Replication is the only default implementation capability. TP/CP/SP require defined partition axes, reductions, state ownership and ordered collectives. Stateful recurrence cannot be split like independent tensor elements. The compiler inserts communication explicitly into rank IR or rejects the plan. A kernel's internal collective requires a declared communicator/effect and rank-consistent order; hidden HCCL calls are outside the contract.

## 6. Foreign kernels: definition independent of implementation

An external operation declares its logical contract once. Any module graph can call it, so both a novel operation and a whole fused layer can be supplied without compiler-source changes.

```inferfabric
kernel RowScale(H: dim) {
    axis T = runtime();
    input x: tensor<bf16>[T, H];
    input scale: tensor<bf16>[H];
    output y: tensor<bf16>[T, H];
    contract {
        semantics = "example.row-scale.v1";
        effects = [];
        aliasing = "none";
        reference = "references/row_scale.json";
        atol = 0.01;
        rtol = 0.01;
        accumulation = "fp32";
        output_rounding = "nearest_even";
    }
}

implementation RowScaleTile for RowScale {
    backend = "tilelang_ascend";
    source = file(path = "kernels/row_scale.py");
    entry = "build_row_scale";
    abi = "inferfabric.kernel.v1";
    toolchain = locked(name = "tilelang_ascend");
    specialize = [H];
    layouts = { x = "contiguous"; scale = "contiguous"; y = "contiguous"; };
    capability {
        target = locked(name = "ascend_local");
        phases = ["prefill", "decode"];
        parallel = "replicated";
        capture = "requires_qualification";
        workspace_bytes = 0;
        workspace_alignment = 64;
    }
}
```

Contract references specify exact equations and golden-vector provenance; a label alone is not proof of semantics. Built-in graph decomposition is the preferred reference where available. A raw implementation can be type-checked against the interface, but the compiler cannot prove arbitrary foreign code computes the stated function, stays in bounds, or respects effects. Qualification is a separate gate. The tolerances above are illustrative acceptance settings, not general BF16 accuracy guarantees.

The same declaration accepts backend IDs `triton_ascend`, `tilelang_ascend`, `ascend_c`, and `pto`. Each adapter defines the source/entry convention and supported targets in its versioned schema. For Ascend C, source may be a C++ file plus declared host-tiling/device dependencies. For PTO, it is source in the selected PTO toolchain's accepted form (for example C++ using PTO interfaces), not an invented universal `.pto` executable format. Triton and TileLang sources may use their native Python syntax during build.

Prefer adjacent source files for editor tooling, imports, testing and ownership. For small kernels, allow `source = inline(text = r###"...raw foreign source..."###);`. Any count of matching `#` delimiters is legal; the chosen closing delimiter must not occur in the body. The lexer produces one opaque token. InferFabric never interprets Python indentation, braces or comments inside it; an adapter receives bytes and a source map. A model still stays readable because implementations can be imported from separate files. No templating or string interpolation is applied to foreign source; specialization values travel as structured build inputs.

All sources, included headers, imported helper modules, compile flags, compiler revisions and native dependencies must be enumerated or captured by a hermetic build manifest. A file outside the declared package/build roots is rejected unless explicitly provided as a locked dependency. Source content is hashed. Formatting the DSL cannot reformat embedded kernel code.

### Build adapter and serving boundary

The adapters are proposed integrations, not a claim that upstream exporters already share an ABI. Triton-Ascend documents compilation to device object code and a separate runtime driver. TileLang-Ascend provides a distinct compilation/runtime stack. PTO is a tile-oriented virtual ISA. These facts motivate separate adapters, rather than treating Python JIT wrappers or PTO source as interchangeable shared libraries. See the primary references below.

An adapter receives a structured request: contract ID, source digest, entry, compile-time parameters, bounded shapes, dtypes/layouts, exact target, and toolchain lock. It returns diagnostics or a bundle containing host launcher, device objects, metadata, workspace requirements and dependency digests. The builder may use Python; the serving process must remain Rust plus native libraries, with no Python interpreter, PyTorch tensors or Python JIT driver required to launch a kernel. If a pinned backend cannot export that form, it is unsupported for native serving until an adapter is implemented. No silent fallback to Python.

The new `inferfabric.kernel.v1` ABI is separate from today's fixed `inferfabric_acl_*` layer API. It uses fixed-width POD descriptors and a versioned function table: inspect, prepare, enqueue, release. Tensor descriptors include data pointer, allocation byte bound, shape, strides, dtype, device, alignment and access mode. Enqueue receives the runtime-owned stream, buffers, workspace, validity/context data, and scalar arguments; it enqueues work without global synchronization or allocations. Completion is established by a runtime event on that stream; other streams require an explicit dependency contract. Ownership survives completion. No STL objects, Rust layout types or exceptions cross the ABI.

Prepare runs before readiness, validates the chosen specialization and reports workspace/tiling requirements. The runtime allocates stable storage before capture. Enqueue cannot compile, allocate, mutate weights, select an unqualified variant, or perform hidden host I/O. Capture support is recorded as unverified/qualified/unsupported for a precise target and specialization; it is not established by an author's boolean. CPU mock adapters test the protocol and failures, not device math.

## 7. Parsing and a formal syntax DOM

Use a hand-written lexer, recursive-descent declaration parser and Pratt expression parser in Rust. This offers explicit recovery and diagnostics for a small grammar without executing a host language. Keep lossless tokens/trivia and span-bearing tree nodes in `inferfabric-syntax`; put name/type/effect checking in separate passes. An implementation spike should compare an arena CST against `rowan`; choose an arena initially unless measured editor needs justify the dependency.

Pipeline:

```text
source -> tokens/trivia -> lossless CST -> typed AST
       -> resolved modules -> typed graph + effects
       -> checkpoint binding -> target/layout lowering
       -> kernel selection -> native bundle -> runtime qualification
```

The AST is a tree of definitions and expressions, not JSON strings in a field map. Required node families: File, Version, Import, Definition, ConfigParam, AxisDecl, PortDecl, StateDecl, InstanceDecl, StackDecl, Section, Field, Graph, Let, Yield, Name/Member/Call/Index/List/Record/Literal/Unary/BinaryExpr, TypeRef, TensorType, TupleType and ErrorNode. Each has `NodeId`, `FileId` and half-open UTF-8 byte span. Retain ordered fields and duplicates for diagnostics rather than overwriting them in a map. Resolved symbol IDs and inferred types live in side tables, not source text.

Example DOM fragment for `let g = tensor.linear(x = x, weight = gate, accumulate = "fp32");`:

```text
Let(binding: g)
  Call
    callee: Member(Name(tensor), linear)
    NamedArg(x, Name(x))
    NamedArg(weight, Name(gate))
    NamedArg(accumulate, String("fp32"))
```

The syntax tree can be exported as versioned `inferfabric.syntax.v1` JSON for inspection. Semantic IR uses a different schema and cannot be confused with a parsed tree. Parse success does not establish name resolution, checkpoint validity, target support or executability.

### Normative syntax core (EBNF)

This grammar defines the proposed v1 surface. Contextual schemas below further restrict valid constructs. `{ X }` means repetition, `[ X ]` optional; quoted strings are terminals.

```ebnf
file         = "language", "inferfabric", integer, ";", { import | definition } ;
import       = "import", string, "as", ident, ";" ;
definition   = defkind, ident, [ parameters ], [ "for", path ], block ;
defkind      = "model" | "module" | "kernel" | "implementation" | "plan" ;
parameters   = "(", [ configparam, { ",", configparam }, [ "," ] ], ")" ;
configparam  = ident, ":", type, [ "=", expr ] ;
block        = "{", { member }, "}" ;
member       = binding | port | state | instance | graph | section | field ;
binding      = ("const" | "axis"), ident, "=", expr, ";" ;
port         = ("input" | "output" | "param"), ident, ":", type, ";" ;
state        = "state", ident, ":", type, block ;
instance     = ("module" | "stack"), ident, "=", expr, ";" ;
section      = ident, block ;
field        = ident, "=", expr, ";" ;
graph        = "graph", "{", { let }, yield, "}" ;
let          = "let", pattern, "=", expr, ";" ;
pattern      = ident | "(", ident, ",", ident, { ",", ident }, ")" ;
yield        = "yield", named, { ",", named }, ";" ;
named        = ident, "=", expr ;
expr         = binary ;
binary       = unary, { binop, unary } ;
unary        = [ "-" | "!" ], postfix ;
postfix      = primary, { ".", ident | "(", [ args ], ")" | "[", expr, "]" } ;
primary      = ident | integer | decimal | string | rawstring | "true" | "false"
             | "(", expr, ")" | list | record ;
args         = named, { ",", named }, [ "," ] ;
list         = "[", [ expr, { ",", expr }, [ "," ] ], "]" ;
record       = "{", { field }, "}" ;
type         = namedtype | "(", type, ",", type, { ",", type }, ")" ;
namedtype    = path, [ "<", typearg, { ",", typearg }, ">" ],
               [ "[", expr, { ",", expr }, "]" ] ;
typearg      = type | integer ;
path         = ident, { ".", ident } ;
binop        = "*" | "/" | "%" | "+" | "-" | "==" | "!="
             | "<" | "<=" | ">" | ">=" | "&&" | "||" ;
```

Pratt precedence, highest first: postfix, unary, multiplicative, additive, comparisons, `&&`, `||`. Arithmetic/logical operators are left associative; comparison chaining is illegal. There is no assignment expression. All calls use named arguments; positional calls are syntax errors. Type parsing is a distinct context: `<`/`>` delimit generic arguments there; a bare type argument such as `T` resolves to a dimension or type as required by the named constructor. A foreign resource input uses a capability type such as `Update<DeltaState>` or `Read<DeltaState>`; the declared state schema supplies the concrete dimensions and layout. This is distinct from a mutable ordinary tensor pointer. Dimension expressions belong in tensor brackets; v1 generic resource arguments are names or integer literals. No `>>` token is defined.

Integers are decimal digits with optional internal single underscores; no implicit octal/hex. Decimal literals include a decimal point or exponent; exponent signs belong to the numeric token. Signs elsewhere are unary operators. Numeric literals are retained exactly in AST; typed conversion checks range and rejects non-finite values. Raw string delimiters follow the rule in section 6. Reserved keywords cannot be identifiers. Lexer longest-match applies to operators and distinguishes strings from comments. Input/file/node/recursion limits and bounded expansion prevent pathological memory use.

Contextual validation makes generic blocks precise: `model` permits label, const/axis, ports, instances, weights and exactly one graph; `module` permits config parameters, const/axis, ports, state, instances and exactly one graph; `kernel` permits config parameters, axes, ports and exactly one contract; `implementation ... for` permits only registered implementation fields and capability; `plan ... for` permits only target, prefill/decode, serving and execution. A state block permits only its resource-policy fields. Other members fail. Backend-specific fields are namespaced inside a schema-checked `options` record, never arbitrary ignored fields. `for` is required only on implementation/plan and forbidden elsewhere.

Single-output calls are tensor expressions; multi-output calls return tuples. Scalar arithmetic applies only to configuration/dimension expressions, never implicitly to tensors. A `yield` must assign every declared output exactly once by name. `RequestContext` is a compiler-defined readonly resource type; `runtime`, `checkpoint`, `repeat`, `run`, `share`, `read`, `update`, `locked` and `target_manifest` are typed intrinsics restricted to their declared context. Intrinsic spelling alone cannot bypass type checking.

## 8. Semantic graph, diagnostics and checkpoint contracts

Passes run in an explicit order: parse; collect/import symbols; evaluate bounded constants; instantiate modules; resolve ports and shapes; validate state/effects; bind weights; validate target/layouts; choose implementations; emit artifacts. Diagnostics distinguish each stage and include the call/instantiation chain.

An error should say, for example: `E2204: gate has shape [6144, 2048], but linear.weight requires [6144, 4096]`, underline the argument, and point to both parameter and dimension declarations. A duplicate binding points to both locations. A state-effect conflict identifies both calls and the instance path. Unsupported code generation reports requested shape/dtype/target and rejected candidate reasons. No generic “invalid model” for these cases.

Checkpoint mapping is a versioned package resource independent of the parser. A declarative mapping maps stable parameter paths to exact tensor names or bounded indexed templates, declares expected dtype/shape, approved transforms, tied storage and explicitly excluded namespaces. Required tensors bind exactly once unless sharing is declared. Preserve existing safetensors bounds, index validation, and path controls. Arbitrary mapping Python is not executed during model checking.

Semantic graph nodes carry qualified operation ID, contract version/digest, typed operands/results, attributes, instance path, state/effect edges, and source spans. They no longer carry `Layer::Full/Delta` as the only extension points. Existing specialized Qwen blocks may remain a fusion optimization after exact pattern/contract validation. Other checked graphs lower through a general call sequence. An opaque foreign node is never expanded into a fabricated built-in layer.

Formatting/comment changes do not change the semantic hash. Hash canonical resolved graphs, parameters/mappings, state schemas, target/layout/parallel plans, implementation choices, raw source dependencies, compiler/adapter/toolchain locks and qualification constraints. Separate parse, typed-graph and executable-bundle versions. Kernel selection is explicit by implementation ID or by unique matching policy; ambiguous candidates fail. No filesystem-order preference. A target-qualified reference implementation may be an explicit fallback; never substitute different mathematics or a mock backend.

## 9. Migration and what remains deliberately unsupported

Keep legacy parsing behind `--legacy-dsl`; do not autodetect based on failed v1 parsing. Supply `inferfabric migrate-dsl` to produce a v1 model plus plan and a report of resolved Qwen assumptions. It must not overwrite input by default. Checkpoint conversion retains the versioned Qwen math contract and existing tensor binding checks.

Initially, v1 graphs in the exact Qwen subset adapt to the existing `Spec`/canonical-plan path. General modules can parse/check before general native lowering exists, but compilation reports a precise unsupported-lowering error. Parsing or CPU mock success never grants native serving readiness.

After general graph execution is available, migrate built-ins to the same module/kernel contracts used by user code. Retire legacy only after fixtures and equivalent graph/tensor bindings demonstrate migration correctness. Do not remove canonical validation before its typed/effect/layout replacement is ready.

Deferred: dynamic tensor control flow, unbounded specialization, arbitrary host callbacks, automatic custom-kernel differentiation, transparent state resharding, automatic distributed custom-layer sharding, a full IDE language server, and claims of numerical correctness without independent references. These can extend the language through explicit versions; opaque strings are not an extension mechanism for core semantics.

## 10. Primary references and design limits

Reviewed 2026-09-08. These describe upstream systems, not tested InferFabric adapters. Pin source revisions and toolchain versions during implementation; documentation on moving branches is not an artifact lock.

- [Triton-Ascend architecture](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/architecture_design_and_core_features.md): separates compiler output from the driver and documents Ascend lowering.
- [TileLang-Ascend](https://github.com/tile-ai/tilelang-ascend): its own Ascend backend and kernel authoring/build stack require an adapter.
- [PTO ISA](https://github.com/hw-native-sys/pto-isa): tile-oriented virtual ISA and toolchain-specific programming surface; not a universal native plugin ABI.
- Local implementation anchors: [parser](../crates/inferfabric-dsl/src/lib.rs), [model](../crates/inferfabric-model/src/lib.rs), [bound graph](../crates/inferfabric-compiler/src/bound.rs), [lowering](../crates/inferfabric-compiler/src/lower.rs), [native FFI](../crates/inferfabric-native/src/ffi.rs).

No NPU execution is required to review this design or implement its lexer, syntax DOM, semantic checks and mock ABI tests. Native numerical, graph-capture and distributed qualification remain separate work that requires future machine time.


## 11. Physical planning and binary execution contract

The user-facing `plan` declaration supplies constraints; a compiler-generated `PhysicalPlan` fixes the execution DAG. Adopt the pipeline and loader contract in [physical planning](physical-planning.md), with a complete [IR catalog](ir-reference.md). Logical operations and state effects are separated from selected kernels, placement, memory offsets and communication.

Planning includes specialization, simplification, implementation/fusion/layout selection, partitioning, scheduling, workspace reconciliation and lifetime allocation. Memory reuse creates ordering dependencies. Binary emission follows verification and freezes those choices. `execute` binds inputs, state, allocations and process-local launch resources; it cannot re-elaborate the model or silently select another implementation. Future JIT creates a new verified bundle through this same boundary.

The local commands `plan`, `explain`, and `execute` currently consume the CPU logical JSON/physical binary formats. This establishes the planning/execution boundary before the new text parser exists. It does not implement all four foreign-kernel adapters or native Qwen binary execution. Preserve these scope distinctions during frontend and native integration.

### Full-model inspection implementation (2026-09-08)

The checkpoint-bound Qwen3.5-2B mathematical plan now has a self-contained, offline visual inspector (`explain-model`). It provides a 24-layer overview, exact operator DAG drill-down, cross-layer SSA navigation, weight/state bindings, search and logical memory formulas. The [full-model reference](validation/2026-09-08-qwen-plan/README.md) records 640 operators. This is separate from the executable CPU physical-plan viewer; native physical schedule/arena export remains follow-up work.
