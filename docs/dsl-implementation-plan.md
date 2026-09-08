# InferFabric DSL v1 implementation plan

Status: proposed, 2026-09-08. Companion specification: [language design](dsl-language-design.md). This is an implementation plan, not a claim that the new parser or foreign-kernel runtime exists. No NPU runs or remote deployment are authorized for this work.

## 1. Outcome and acceptance boundary

Deliver a readable, versioned language with a formal syntax DOM, reusable typed modules, explicit state/effects, and a kernel extension interface. A user must be able to add a new graph-defined layer by adding package files; a raw kernel using an installed backend adapter must not require editing an architecture enum or rebuilding InferFabric's compiler.

Do not mistake parser completion for model execution support. Track four independent outcomes:

| Outcome | Evidence | Machine time |
|---|---|---|
| Language support | Parse, format, AST and semantic conformance suites | CPU only |
| Compiler integration | Qwen equivalence, binding validation, deterministic general graph lowering | CPU only |
| Adapter integration | Structured build protocol, native ABI checks, local mock adapter; toolchain builds where available | CPU; no device launch |
| Native qualification | Golden numerical tests, lifetime failures, capture replay and target-specific behavior | Deferred until explicitly authorized |

Triton-Ascend, TileLang-Ascend, Ascend C and PTO are all required extension routes. Their adapters may reach different maturity levels; report each separately. An unavailable compiler or non-exportable Python runtime must be an explicit unsupported adapter state, not a successful native integration.

## 2. Intended repository structure

```text
crates/
  inferfabric-syntax/         # lexer, lossless CST, typed AST, source maps
  inferfabric-dsl/            # loading, resolution, elaboration, diagnostics
  inferfabric-model/          # model-independent types and contract identities
  inferfabric-ir/             # graph, effects, layouts and rank programs
  inferfabric-compiler/       # binding, lowering, selection and artifacts
  inferfabric-kernel-api/     # native ABI descriptors and bundle validation
  inferfabric-cli/            # parse/fmt/check/migrate/compile entry points
stdlib/
  tensor/                    # versioned primitive schemas
  qwen35/                    # inspectable modules, mapping, state contracts
examples/dsl-v1/
  qwen35-2b.inferfabric
  qwen35-tp2.inferfabric
  gated-mlp.inferfabric
  custom-row-scale/
    module.inferfabric
    kernels/                 # four backend source implementations
    references/              # tiny, provenance-bearing golden data
    inferfabric.lock
adapters/
  triton-ascend/
  tilelang-ascend/
  ascend-c/
  pto/
```

Keep the syntax crate independent of CANN, Python, checkpoint loading, and device libraries. Keep adapters out of ordinary parse/check dependencies. Reuse existing low-level bounds checks and native ownership wrappers rather than copying them. Do not modify vendored vLLM to introduce the DSL.

## 3. Phase A — freeze executable language fixtures

Tasks:

- Turn the design's Qwen model, plan, GatedMLP, stateful module and external-kernel examples into candidate v1 fixture files, with all required imports and standard declarations.
- Publish an exact keyword/token table, EBNF, contextual field schemas, numeric rules and precedence table alongside fixtures. Resolve any discrepancy before implementing interpretation.
- Add a standard-module contract inventory: ports, configuration defaults, parameters, state, shape equations, rounding, and checkpoint mappings. Keep Qwen-specific semantics in the Qwen package.
- Record version policy for language, syntax JSON, graph IR, kernel ABI, standard packages, and native bundles.

Exit gate: every syntactic feature used by a fixture has a grammar production and every field has a schema. No inference of semantics from a display name. Review the Qwen source for readability with and without its expansion view.

This phase does not generate a device implementation or promise all declared Qwen modules already exist.

## 4. Phase B — lexer, parser, syntax DOM and formatter

Implement a Rust lexer with byte spans, preserved whitespace/comments, numeric literals, JSON strings, raw foreign-source strings and explicit error tokens. Add bounded token/node/depth accounting.

Implement a recursive-descent declaration parser and a Pratt expression parser. Collect errors with recovery at `;`, `}` and declaration starts; represent malformed constructs as error nodes. Recovery must guarantee progress. Avoid parsing nested blocks using regular expressions or losing duplicate fields in maps.

Add typed CST-to-AST accessors and versioned JSON export. AST nodes must retain declaration ordering, named arguments, type dimensions, graph statements, state policy and opaque raw source. Source IDs resolve through a central source database. JSON spans use UTF-8 byte offsets; editor line/column conversion is a separate function with Unicode tests.

Implement `inferfabric parse FILE --emit ast`, `inferfabric fmt FILE --check`, and explicit formatting writes. These command names are proposed, not currently available. AST output includes unresolved names and is labeled non-executable. Formatting must be idempotent and preserve foreign source byte-for-byte.

Tests and exit gate:

- All valid fixtures parse into checked-in AST snapshots; snapshots complement structural assertions rather than replacing them.
- Invalid fixtures cover missing delimiters/semicolons, duplicate argument spelling, malformed strings, ambiguous-looking operators, chained comparisons, malformed tensor types and truncated files.
- Round-trip property: lossless CST reconstruction equals the original bytes.
- Formatting property: formatting twice equals formatting once, and reparsing preserves structural AST except trivia/spans.
- Property/fuzz tests establish no panic/hang, valid spans and bounded recovery for arbitrary UTF-8, deep nesting and delimiter-heavy embedded source.
- Parser tests run without importing Python, accessing checkpoints, or resolving network dependencies.

Deliver parser and formatter first; do not bundle kernel execution into this change.

## 5. Phase C — resolution, types and effect-aware elaboration

Implement explicit imports, lockfile-backed package resolution, duplicate/cycle diagnostics, lexical scopes, module parameter binding and stable instance IDs. Resolve declaration references in a second pass; reject forward graph values and recursive module expansion.

Add bounded constant evaluation and `repeat` expansion. Track definition span plus instantiation stack so an error inside the sixth repeated layer can be traced to its call site. Use checked arithmetic, expansion limits and a deterministic iteration order.

Introduce model-independent types, symbolic dimensions and constraints. Distinguish runtime capacity axes, valid token metadata and configuration parameters. Validate shapes, dtype, layout contracts, named ports, output arity and explicit casts/broadcasts. Unknown shape proofs remain errors at executable lowering, not unchecked assumptions.

Lower state access to resource IDs and effect edges. Check read/write conflicts, request-scoped generation ownership, reset/transfer requirements, and unknown aliases. Kernel contracts become ordinary graph call targets; they do not need an `Op::MyNewLayer` enum variant.

Tests and exit gate:

- A novel graph module with a name absent from Rust source resolves and elaborates without compiler edits.
- Shape mismatch, duplicate binding, accidental state sharing, unordered writers, unsupported broadcast, missing output and recursive expansion all fail with primary/secondary spans.
- Different formatting yields the same canonical graph digest. Changing an operation contract, weight mapping or state schema changes it.
- TP=2 on a replicated-only custom kernel is rejected unless the plan explicitly uses replication with compatible surrounding layout.
- No native code is loaded by semantic checking.

## 6. Phase D — Qwen standard package and legacy migration

Extract the existing Qwen mathematical definition into inspectable standard-module graphs and a strict checkpoint mapping. Retain small compiler intrinsics for primitive semantics; do not hide the whole model behind a `contract` string again.

Provide `inferfabric migrate-dsl OLD --output-dir NEW` and `--legacy-dsl` routing. Preserve the old parser as a separate entry point while migration is active. A malformed v1 file must never silently enter the legacy parser. Emit the old 24-layer schedule as explicit bounded composition with stable weight/state paths.

Use a deliberately limited adapter from the supported v1 Qwen subset to today's `Spec` and canonical bound plan. Unsupported custom graphs get a lowering diagnostic until Phase F; they cannot be coerced into `Full` or `Delta`. Preserve canonical equality protection during this transitional phase.

Tests and exit gate:

- For each current example, legacy and migrated inputs agree on dimensions, layer order, mesh, buckets, parameter identities, state schemas and logical operations after normalization.
- Compare to separately captured expected Qwen contracts, not only two paths sharing the same generator.
- Checkpoint fixture tests retain strict binding, tying, ignored-namespace policy, shape/dtype checks and overflow/path protection.
- Tests explicitly protect one-plus RMSNorm, Q/gate interleave, RoPE parameters, recurrence orientation/precision and convolution history.
- Build and run only the mock compiler/scheduler path locally. Report native parity as untested until device qualification.

## 7. Phase E — foreign-operation package and ABI, with CPU mocks

Introduce `inferfabric-kernel-api` and its matching C header. Define size/version fields, dtype/layout enums, tensor bounds, scalar encoding, stream/event contract and inspect/prepare/enqueue/release function table. Add bundle metadata, target/toolchain locks and dependency digests.

Write a tiny local C/C++ CPU mock plugin implementing the ABI. It should exercise output writes, workspace requests, invalid metadata, asynchronous ownership simulation, prepare failures and release ordering. It must not link CANN or open devices. Build it only in a designated local test directory.

Implement an adapter process protocol using structured JSON plus binary artifacts. Never construct arbitrary shell commands from DSL text. Parsing/checking only reads manifests. The separate `kernel build` command explicitly runs a selected build adapter with time/resource bounds and declared inputs. Capture diagnostics and map foreign-source line numbers back to file/inline spans.

Tests and exit gate:

- Add a new external operation plus CPU mock implementation entirely as package data; compile it into a generic call node without editing Rust layer enums.
- Reject wrong ABI versions, missing symbols, undeclared outputs/effects, out-of-bounds descriptors, overlapping forbidden buffers, stale state generations, incompatible layouts and corrupted bundle hashes.
- Verify explicit target/candidate selection, ambiguity errors and absence of automatic mock fallback.
- A compiler crash yields a build diagnostic and does not create a valid bundle or corrupt an existing cache entry.
- Stateful resources appear in transfer manifests or produce a declared unsupported-PD error.

An ABI-conforming mock proves protocol integration, not the safety or accuracy of arbitrary device code.

## 8. Phase F — general graph lowering and runtime integration

Replace fixed-layer execution as the only route with a typed call-sequence program supporting buffers, views, parameters, state accesses, operation calls, events and explicit collectives. Add verification of lifetimes, offsets, layouts, effects and output coverage before linking. Retain the Qwen optimized block path as a verified fusion of the same semantic graph.

Migrate native execution incrementally: generic buffer planning; generic invocation; state/context plumbing; optional fusion; graph preparation/capture metadata. Preserve request admission, cancellation and storage ownership. Keep the vLLM Rust frontend boundary unchanged.

Exit gate on CPU:

- A model mixing a standard module and a user-defined external operation lowers without rebuilding the compiler.
- Generic mock execution matches a small independent Rust reference graph, including request reuse and effect ordering.
- Mutating serialized operation/shape/effect metadata is rejected by the verifier.
- Requested native execution fails before readiness if any call lacks a suitable native implementation or qualification record.

Device execution of this new path remains deferred. Do not remove the current tested native path merely because generic mock execution passes.

## 9. Phase G — implement each real backend adapter

For each backend, pin a specific revision and conduct a build/export feasibility check. Do not assume a vendor Python launcher has an equivalent supported C ABI. Preserve Python-free serving by making the export bridge an explicit adapter responsibility.

| Adapter | Required local build deliverable | Key unresolved integration check |
|---|---|---|
| Triton-Ascend | Compile raw kernel source and specializations; package device objects, launch metadata and native shim | Eliminate dependency on Python/Torch launch objects in the serving path |
| TileLang-Ascend | Compile the selected entry and declared dependencies; export native launcher and workspace contract | Match generated host/tiling code and asynchronous launch to the kernel ABI |
| Ascend C | Build user host tiling and device code with the pinned CANN toolchain; expose versioned function table | Preserve stream ownership, shape-specific workspace and resource lifetime |
| PTO | Build the pinned toolchain's accepted source and integrate its launcher through the same ABI | Exact PTO version/SoC/compiler compatibility and binary packaging |

Use one RowScale contract with independently authored reference vectors across all four adapters. Include a second stateful example with explicit request validity metadata; a stateless multiplication example alone does not prove layer extensibility. Add a multi-operation custom layer composed from these calls to show that kernel and layer are not synonyms.

Local adapter checks may run only when the necessary toolchains are available without contacting a device. Mark missing-toolchain jobs skipped with a reason. Offline compilation must not autotune, benchmark, query live device properties or trigger an implicit JIT warmup. All target information comes from a supplied manifest.

Exit gate: per adapter, a reproducible build manifest, validated native bundle, ABI/link inspection, and explicit unresolved runtime qualification. An adapter that cannot export a Python-free launcher remains experimental/unavailable; document the concrete missing interface instead of claiming completion.

## 10. Phase H — later device qualification, not part of current execution

Only after machine time is explicitly provided: compare each real kernel against independent golden outputs; verify padding/tail cases, NaN/Inf policy, rounding, state continuation and invalid parameter rejection. Compare model intermediate tensors/logits, not just generated text.

Exercise graph capture/replay for all declared buckets, stable addresses, repeated inputs, changed validity metadata, state reset and cancellation. Verify no allocations, compilation, hidden synchronization or Python execution on enqueue. Check replicated execution before qualifying any sharding rule. PD tests must include every custom resource and version mismatch rejection.

Record exact source/toolchain/bundle hashes and target versions. Qualification records are bound to those identities and shapes; a source, compiler, layout, or ABI change invalidates relevant evidence. Missing machine time cannot be replaced with a “passed” mock test.

## 11. Reviewable delivery slices and ordering

1. Grammar/schema fixtures and syntax crate skeleton.
2. Lexer/CST/AST with conformance tests.
3. Formatter, AST CLI and diagnostics.
4. Module elaboration and type/shape/effect checking.
5. Qwen standard package, legacy migration and compiler-subset adapter.
6. Kernel package schema, native ABI and CPU mock adapter.
7. General graph lowering and CPU mock execution.
8. Four separately reviewable backend adapter integrations.
9. Deferred per-target qualification and eventual legacy retirement.

Dependencies: 1 -> 2 -> 3/4; 4 -> 5 and 6; 5/6 -> 7; 6 -> adapter build work in 8; 7/8 -> 9. Work can be split later, but this plan does not dispatch agents or create background tasks.

Each slice should keep the current workspace and frontend tests passing. Use the repository's Rust formatting and lint checks, targeted new tests, and local mock integrations. Do not add hardware runs to default `cargo test`. Native qualification commands must be explicitly separate.

## 12. Immediate next action and review decisions

Start with Phases A and B after design review. The next code change should produce a formal parse tree and formatter for v1 fixtures, not a rewritten native runner. Keep the examples under a v1 directory so they cannot be mistaken for inputs accepted by the current parser.

Proposed decisions are already made in the design: braces and semicolons; named arguments; bounded compile-time composition; lossless CST plus typed AST; explicit state; external source files preferred; a separate plan; adapters behind one versioned native ABI. Review may revise these before the grammar fixtures are frozen.

The major technical uncertainties are native export from the selected Triton/TileLang toolchains, exact PTO integration, generic graph/runtime migration and custom-state qualification. None blocks designing or implementing the parser and syntax DOM. No schedule estimate should hide these as ordinary parser work.
