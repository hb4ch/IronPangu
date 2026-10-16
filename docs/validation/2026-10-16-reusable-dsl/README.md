# Reusable DSL: full 92B NPU qualification — 2026-10-16

**Passed:** the 107-line modular OpenPangu V2 92B definition expands to the
same 2,109 primitive operations as the previously qualified 2,221-line source.
The complete 46-layer decoder ran on eight Ascend NPUs with TP8/EP8. All 33
generated tokens, including EOS, match the pinned offline reference. The entire
results JSON is byte-identical to the previous flat-DSL run, including recorded
selected logits and top-five values.

Implementation milestones: `6142968` (language and expansion) and `7ed86e9`
(compact model and regression tests).

## Authoring and expanded plan

The [model source](../../../examples/openpangu/92b-tp8-ep8.inferfabric) uses five
file-local modules, a 46-iteration compile-time fold, static branch selection,
and scoped checkpoint bindings. Modules expand before graph validation and
initialization planning; no runtime module interpreter is involved.

- [Complete expanded execution IR](execution.json): every operation, dependency,
  concrete binding, hygienic instance path, source byte span and expansion stack.
- [Graphviz diagram](execution.dot): all expanded operations and edges.
- [Migration comparison](migration-comparison.json): exact primitive operation,
  dependency and attribute equality, identical output node, all eight
  initialization binaries byte-identical, and valid source spans.
- [Artifact hashes](artifacts.sha256): executable, custom kernel, execution and
  initialization programs, configuration and tokenizer provenance.

The native executable is unchanged, with SHA-256
`b1f076aadda7d0c2074091dde70e50f6cf1af4953cb06d74204aefa198545fd5`.
The execution binary changes because hygienic names replace manually numbered
node names. Source metadata is retained in JSON and does not enter binary encoding.

## Actual outputs

| Prompt | Complete answer before EOS | Tokens including EOS |
|---|---|---|
| `1+1=?` | `1 + 1 = 2` | 8 |
| `What is the capital of France?` | `The capital of France is Paris.` | 8 |
| `请用一句话介绍北京。` | `北京是中国的首都，拥有悠久的历史、丰富的文化遗产和现代化的城市风貌。` | 17 |

[Raw results](results.json) and [reference comparison](reference-comparison.json)
record the complete generations. The reference is the saved pinned result from
the [earlier full-decoder qualification](../2026-10-15-openpangu-decoder/README.md),
not a newly rerun Torch reference. Comparison with the
[previous flat-DSL run](../2026-10-15-dsl-execution/README.md) also checks the
complete recorded results byte for byte.

## Validation and reproduction

- Workspace Rust tests and Clippy with warnings denied passed.
- All six frontend tests passed.
- Expansion tests cover reuse, scoped bindings, hygienic state identity, named
  arguments, static branches, zero-iteration folds, recursion, invalid inputs,
  and bounded expansion work/nesting.
- The model regression checks all 2,109 operations against the saved flat graph.
- Native `token_plan_test` passed all eight rank bindings and seven negative cases.
- Rust was cross-compiled locally under WSL. Native validation and NPU execution
  ran only inside Docker `inferfabric-92b` on `7.156.99.58`.

```sh
# Local WSL:
bash scripts/cross-compiler.sh
# Transfer the compiler and source to Docker; inside Docker:
.deploy/inferfabric-compiler compile-token \
  examples/openpangu/92b-tp8-ep8.inferfabric /models/92B reusable-dsl-plan \
  --allow-stale-index-size
bash scripts/qualify-openpangu-decoder.sh \
  /models/92B reusable-dsl-plan decoder-reusable-dsl-qualified
python3 scripts/compare-openpangu-decoder.py \
  decoder-reusable-dsl-qualified/results.json \
  docs/validation/2026-10-15-openpangu-decoder/reference/results.json comparison.json
```

Choose fresh output directories when repeating the commands. The stale-index
option applies to this previously audited checkpoint; shape and shard checks
remain enabled. Remote artifacts are under
`/data/p00603624/inferfabric/openpangu92/reusable-dsl-plan` and
`/data/p00603624/inferfabric/openpangu92/decoder-reusable-dsl-qualified`.

## Scope

This qualifies the complete model with TP8/EP8, one sequence, serial token
prefill, greedy decoding and a 512-token context bound. Long-context indexing,
continuous batching, full-model ACLgraph, Rust serving integration for this
program format and production performance remain separate milestones. Exact
results on these three prompts do not establish broad model quality or equality
of every vocabulary logit.
