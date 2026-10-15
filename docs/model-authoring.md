# Model authoring and compiled implementations

There are two equal authoring paths. A versioned DSL parses to a span-bearing AST and elaborates to `inferfabric_authoring::ModelDefinition`. A compiled Rust package implements `ModelAuthor` and returns that same definition. Both undergo the same semantic checks and use the same checkpoint binding, initialization lowering and physical planner. A compiled author can use normal loops/functions; the parser never evaluates host code. Other compiled languages can emit the documented `inferfabric.model` version-1 JSON envelope, but Rust is the provided SDK.

The legacy unversioned `key = value` parsers are removed, including the separate OpenPangu parser. No automatic fallback exists. Checked-in Qwen, OpenPangu and frontend fixtures use `language inferfabric 1;` with separate model and plan declarations. Existing compiled runtime formats are not invalidated just because source syntax changed.

## Supported language subset

See [graph example](../examples/authoring/residual.inferfabric) and [92B definition](../examples/openpangu/92b-tp8-ep8.inferfabric). Graph models currently support static f32 inputs, constants, persistent state, add/mul/relu/matmul, named outputs and simultaneous state updates. The CPU planner validates shapes/effects and emits source-independent executable plans. Existing Qwen/OpenPangu contracts are explicit adapters to their established model binding, not claims that their complete computation graphs are now authored in the new grammar.

The lexer understands quoted strings, comments, lists and calls; declarations and members carry byte spans. Unknown fields, duplicate declarations, invalid versions, unsupported operations and excessive input/nesting are rejected. This is an implemented subset of the larger language design: reusable/imported modules, lossless CST/formatting, symbolic shapes, arbitrary foreign operations and general NPU graph lowering remain unsupported. Unsupported constructs fail rather than reverting to the retired parser.

## Compiled-language route

[The compiled model example](../crates/inferfabric-authoring/examples/model.rs) has no DSL dependency. It emits either the same residual graph as the DSL example or the OpenPangu TP8/EP8 definition:

```bash
cargo run -p inferfabric-authoring --example model -- residual residual.ifmodel.json
cargo run -p inferfabric-authoring --example model -- openpangu92 openpangu92.ifmodel.json
cargo run -p inferfabric-cli -- model-source examples/authoring/residual.inferfabric dsl.ifmodel.json
cargo run -p inferfabric-cli -- plan residual.ifmodel.json residual.ifplan --dump-ir ir
```

The compiled OpenPangu artifact can replace the DSL argument to `export_openpangu_load_plan` and `plan-openpangu`. The Qwen compiler/frontend source readers likewise accept the versioned compiled model artifact. Serialized model definitions are authoring interchange, not executable foreign code. The current OpenPangu profile specifies topology; its architecture still comes from the validated checkpoint, and its decoder remains explicit native code.

## Debugging and tuning plans

`plan-debug` disables dead-node removal and constant folding while retaining validation. `plan --dump-ir` records logical, typed, optimized and physical stages. `explain --html` visualizes the actual physical plan. `tune-plan` accepts a map from step name to registered kernel variant, verifies that operation semantics are unchanged, revalidates the whole physical plan and emits a new binary:

```bash
inferfabric plan-debug examples/authoring/residual.inferfabric debug.ifplan --dump-ir debug-ir
# overrides.json: {"projected":"matmul_blocked_f32_v1"}
inferfabric tune-plan debug.ifplan overrides.json tuned.ifplan
inferfabric explain tuned.ifplan --html tuned.html
```

Compiled callers use `compile_with_options`, `override_kernels` and `Executor::bind_implementation`. `CompiledKernel` supplies actual compiled Rust math for a named existing operation; the executor retains input/output shape and finiteness checks and commits state only after successful execution. These callbacks are explicitly installed by trusted compiled code, never loaded implicitly by a plan. Numerical equivalence remains a test obligation. This CPU interface copies inputs and is a debugging/reference seam, not an Ascend performance API.

## Native Ascend C boundary

Model definition must remain independent of implementation language. An Ascend C implementation may replace an operation or a fused subgraph only through the native implementation contract, preserving tensor layouts, effects, workspace, stream ordering and numerical acceptance criteria. The implementation must retain editable source and build provenance, rather than requiring edits to generated code that are lost on recompile. Plan overrides select a separately built implementation; they must not disguise changed model semantics as a backend optimization.

The current 92B runtime remains compiled C++/CANN code with the common initialization executor. The general Ascend C/PTO/TileLang/Triton build adapters and generic NPU execution-plan integration are separate work from the now-working dual model-authoring paths. Do not interpret CPU override tests as proof of Ascend kernels or graph-capture safety.

## Validation

DSL and compiled Rust definitions produce identical model artifacts and executable CPU plans. Tests cover stateful execution, debug planning, implementation overrides, compiled execution callbacks, malformed definitions and explicit legacy rejection. All six Rust frontend tests pass after fixture migration. On the pinned 92B checkpoint, the DSL, compiled author and pre-migration definitions produce byte-identical IFINIT01 programs on all eight ranks. The full 46-layer TP8/EP8 NPU decoder still produces the identical complete three-prompt results JSON (33 tokens including EOS).
