# Shared initialization executor: full 92B regression

The OpenPangu-specific loader is removed. Rust placement lowering now emits shared, typed `InitializationPlan` IR, its readable JSON and portable `IFINIT01` binaries. The 92B decoder executes those programs through the same model-independent executor tested with an injected memory backend. Runtime allocation, file reads, conversions, upload and cleanup no longer depend on OpenPangu or an eight-rank limit.

## Results

- All eight ranks loaded 9,956 parameters each; all uploaded bytes were read back and compared successfully. Source bytes per rank: 12,885,336,434. Resident bytes per rank: 12,945,669,860, unchanged from the earlier loader.
- All three full-model prompts completed through 46 layers and TP8/EP8, with finite residual checks and eight-rank token agreement.
- All 33 tokens, including EOS, matched both the pre-refactor native decoder and the pinned offline OmniInfer reference.
- The entire native `results.json` is byte-for-byte identical to the pre-refactor result at its recorded JSON precision, including selected logits and top-five log probabilities. This does not claim bitwise equality of unrecorded full-vocabulary logits.
- The reference log-probability differences are unchanged: maximum selected-token absolute deltas 0.030322 (arithmetic), 0.030492 (geography), 0.103009 (Chinese).
- Rust workspace tests and Clippy with warnings denied passed. Rust was cross-built locally in WSL. Native C++ builds and all NPU activity ran inside Docker on 7.156.99.58.
- Native executor tests passed for bounded multi-chunk copy/cast, exact readback, budget and source-range rejection, generic rank 17, invalid producers/conversions, nonfinite casts and cleanup after an injected upload failure. All 17 binary-format validation cases passed.

`load/` contains per-rank reports, test output, dependency records and executable/plan hashes. `decoder/` contains the full generation results, executable/config/tokenizer hashes and dependencies. `native-comparison.json` and `reference-comparison.json` record comparisons. The checkpoint, container, operator versions and independent reference are unchanged from the [previous complete decoder qualification](../2026-10-15-openpangu-decoder/README.md).

## Reproduce

Cross-build with `scripts/cross-openpangu-probe.sh` locally and transfer the resulting exporter into the Docker checkout. Then run inside Docker:

```bash
chmod +x .deploy/export_openpangu_load_plan
bash scripts/qualify-openpangu-load.sh /models/92B \
  examples/openpangu/92b-tp8-ep8.inferfabric init-qualification
bash scripts/qualify-openpangu-decoder.sh /models/92B \
  init-qualification/plan decoder-init-qualified
python3 scripts/compare-openpangu-decoder.py decoder-init-qualified/results.json \
  docs/validation/2026-10-15-openpangu-decoder/reference/results.json reference-comparison.json
cmp decoder-init-qualified/results.json \
  docs/validation/2026-10-15-openpangu-decoder/device/results.json
```

`init-qualification/plan/rank-N.init.json` is the readable initialization program; `rank-N.bin` is its executable encoding. `plan.json` remains the descriptive model placement plan. Older IFLOAD01 binaries must be regenerated and fail explicitly if supplied.

## Scope

This replaces the weight-loading infrastructure, not the handwritten decoder's execution DAG. Operator-specific transposes/folding, host transfers, serial prefill, greedy decoding, the 512-token limit, and outstanding full-model ACLgraph/Rust serving integration remain as documented. The generic initialization IR currently supports streamed copy and BF16-to-FP32 conversion; unsupported preparation must gain explicit typed operations rather than model-specific loader branches. Checkpoint fingerprints remain provenance, not cryptographic payload verification. See the [initialization contract](../../initialization-plan.md).
