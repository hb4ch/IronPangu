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

## Native compiled implementation hook

The subsequent native milestone moves dense/shared BF16 SwiGLU behind the shared `inferfabric_kernel_v1` POD ABI. The default separately compiled implementation uses CANN; an explicitly selected shared object can replace it without changing model definition or decoder sequencing. `INFERFABRIC_SWIGLU_LIBRARY` selects the implementation before execution. The qualifier copies the selected library into the run directory and hashes it with the executable; see `native-kernel-artifacts.sha256`.

The final three-prompt 46-layer NPU run through this hook produced a results JSON byte-identical to the previous native baseline. All 33 tokens including EOS also match the independent reference; `native-kernel-reference-comparison.json` records the unchanged log-probability differences. This qualifies the shipped CANN implementation through the ABI, not an arbitrary custom Ascend C implementation.

Native interface tests pass for wrong ABI/contract/target, malformed dtype/descriptor/strides, allocation bounds, illegal aliases, implementation prepare failure, workspace capacity/alignment, retained shared-library lifetime, successful compiled execution and rejection of repeated enqueue. The prepared-instance single-enqueue guard was fixed after a negative test caught its absence; the final full-model run includes that guard.

Build interface tests inside Docker with:

```bash
cmake --build native-build --target compiled_kernel_test kernel_test_impl kernel_bad_impl -j4
native-build/compiled_kernel_test native-build/libkernel_test_impl.so native-build/libkernel_bad_impl.so
```

The full-model qualification command above now snapshots both native decoder and selected kernel library. This initial ABI supports contiguous disjoint tensors, fixed prepared bindings and caller-owned workspace/stream. General native graph lowering, serialized production implementation selection, authenticated build provenance, hermetic Ascend C/PTO/TileLang/Triton adapters and full-model capture remain separate work. See [the implemented native contract](../../model-authoring.md#native-ascend-c-boundary).
