# Complete DSL-driven 92B execution — 2026-10-15

**Passed:** the complete 46-layer OpenPangu V2 92B graph was compiled from
[the DSL](../../../examples/openpangu/92b-tp8-ep8.inferfabric), loaded with
graph-derived initialization programs, and executed on eight Ascend NPUs.
All 33 generated tokens, including EOS, match the pinned offline reference.
The entire results JSON equals the previous native decoder's device results,
including recorded selected logits and top-five log probabilities.

Implementation milestone: `fdb383e`. Native binary SHA-256:
`b1f076aadda7d0c2074091dde70e50f6cf1af4953cb06d74204aefa198545fd5`.
The subsequent `compile-token` CLI produces byte-identical `execution.bin`
and all eight `rank-N.bin` files to this tested bundle.

## Actual outputs

| Prompt | Complete answer (before EOS) | Tokens including EOS |
|---|---|---|
| `1+1=?` | `1 + 1 = 2` | 8 |
| `What is the capital of France?` | `The capital of France is Paris.` | 8 |
| `请用一句话介绍北京。` | `北京是中国的首都，拥有悠久的历史、丰富的文化遗产和现代化的城市风貌。` | 17 |

[Raw results](results.json), [reference comparison](reference-comparison.json),
[complete execution IR](execution.json), [binary/config/tokenizer hashes](artifacts.sha256).
Reference provenance and the earlier native baseline are in
[the full-decoder qualification](../2026-10-15-openpangu-decoder/README.md).
This compares with the saved pinned reference, not a newly rerun Torch reference.

## What the program controls

The 2,109-node program specifies projections, TP gathers, mHC residual paths,
MoME, absorbed MLA query/value math, cache append, learned sinks, attention,
expert routing and owner-local projections, fused quantized activation/down,
weighted reduction, all-reduce, shared experts and final normalization/logits.
The C++ decoder has no layer loop or architecture-forward fallback.
Each rank independently reads the same execution program; all eight ranks'
selected-token consensus is checked during generation.

The generic initializer now binds **9,892 parameters/rank** and allocates
**12,832,550,628 weight bytes/rank**. The previous model-name binder loaded
9,956 parameters and 12,945,669,860 bytes/rank. The 64 unused indexer tensors
are omitted because the short-context graph never references them, saving
113,119,232 bytes/rank. These are weight residency totals, not peak runtime HBM.

## Plan-control experiment

A separate [DSL source](bypass-layers.inferfabric) deliberately removes every
transformer block and connects embedding streams directly to the final merge.
This compiles to seven operations and seven parameters/rank. It is intentionally
**not** the correct model and is not a quality result.

Running the **same native executable** on this program with the arithmetic
prompt changes the first token from **16** (logit 130) to **35994** (logit 17.125).
See [the recorded output](bypass-result.jsonl) and
[initialization/control measurements](initialization-and-control.json).
No native source was changed or rebuilt for this experiment. It confirms that
both loading and inference follow the supplied graph instead of dispatching a
hidden fixed Pangu decoder.

## Validation and reproduction

- `cargo test --workspace`: passed, including typed graph, effect pairing,
  route provenance, parameter shape/dtype/placement and compiled-artifact tests.
- `cargo clippy --workspace --all-targets -- -D warnings`: passed.
- `cargo test --manifest-path frontend/Cargo.toml`: all six tests passed.
- Native `token_plan_test`: 2,109 nodes and all eight ranks' bindings passed;
  seven malformed kernel/binding cases rejected before device allocation.
- `compile-token`: all nine executable program files byte-identical to the
  NPU-tested bundle. Publication tests verify complete output and refusal to
  overwrite an existing bundle.
- Rust compiled locally under WSL; native C++/CANN builds and NPU execution
  performed only inside `inferfabric-92b` on `7.156.99.58`, devices 0–7.

```sh
# Local WSL
bash scripts/cross-compiler.sh
# Transfer the binary/source to Docker, then inside Docker:
.deploy/inferfabric-compiler compile-token \
  examples/openpangu/92b-tp8-ep8.inferfabric /models/92B new-dsl-plan \
  --allow-stale-index-size
bash scripts/qualify-openpangu-decoder.sh /models/92B new-dsl-plan new-dsl-results
python3 scripts/compare-openpangu-decoder.py new-dsl-results/results.json \
  docs/validation/2026-10-15-openpangu-decoder/reference/results.json comparison.json
```

Remote verified artifacts remain at
`/data/p00603624/inferfabric/openpangu92/decoder-dsl-native-final`.
The qualifier snapshots the execution and initialization binaries before running.
No model Python or Torch executes in the native inference path; Python is used
only for tokenization/test orchestration and comparisons.

## Scope

TP8/EP8, one sequence, serial token prefill, greedy decoding, at most 512 total
tokens. All causal keys fit the declared SWA/sparse windows at this bound.
No long-context indexer, continuous batching, full-model ACLgraph, production
latency or 92B Rust serving qualification is claimed. Existing custom SwiGLU
implementation binding remains active. Exact tokens on three prompts are not a
broad model-quality or bitwise full-logit equivalence certificate.
