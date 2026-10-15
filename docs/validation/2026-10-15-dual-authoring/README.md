# Dual authoring and legacy-source removal

The versioned DSL and the compiled Rust `ModelAuthor` example both emitted byte-identical IFINIT01 programs for all eight 92B ranks. They also matched the previously qualified initialization programs. The full 46-layer TP8/EP8 NPU run using the compiled author's programs returned a results JSON byte-identical to [the previous qualified outputs](../2026-10-15-openpangu-decoder/device/results.json): all 33 generated tokens, including EOS, plus recorded logits and top-five log probabilities match. The executable/config/tokenizer hashes are recorded in `artifacts.sha256`; this migration did not change native decoder math.

Rust workspace tests, fmt and Clippy with warnings denied pass. All six frontend tests pass with migrated fixtures. New tests cover DSL/compiled artifact equivalence, physical-plan equivalence, stateful graph execution, debug planning, checked kernel overrides, compiled math implementations, malformed inputs and legacy rejection. Rust cross-compilation ran locally; checkpoint planning and NPU execution ran in Docker on 7.156.99.58.

Reproduce the two definitions locally:

```bash
cargo run -p inferfabric-authoring --example model -- openpangu92 .deploy/openpangu92.ifmodel.json
bash scripts/cross-openpangu-probe.sh
```

Transfer the exporter, model artifact and v1 DSL into the NPU Docker checkout, then:

```bash
.deploy/export_openpangu_load_plan /models/92B examples/openpangu/92b-tp8-ep8.inferfabric authoring-dsl-plan --allow-stale-index-size
.deploy/export_openpangu_load_plan /models/92B .deploy/openpangu92.ifmodel.json authoring-code-plan --allow-stale-index-size
for rank in {0..7}; do cmp authoring-dsl-plan/rank-$rank.bin authoring-code-plan/rank-$rank.bin; done
bash scripts/qualify-openpangu-decoder.sh /models/92B authoring-code-plan decoder-authoring-qualified
cmp decoder-authoring-qualified/results.json docs/validation/2026-10-15-openpangu-decoder/device/results.json
```

The language is a declared v1 subset, not the entire module proposal. OpenPangu remains a checkpoint-bound topology contract with an explicit compiled decoder; general NPU graph lowering and foreign backend adapters are not established by these tests. See [supported interfaces](../../model-authoring.md).
