# Explicit dense-attention window metadata

Date: 2026-10-17. Production prerequisite for SWA recycling; context remains 512.

Previously, dense attention and its metadata producer used maximum model context
minus one as `pre_tokens`, and every dense attention shared a single metadata
buffer. That only represented the DSL window while context was bounded below it.
It would incorrectly expand SWA attention when enabling longer contexts.

The compiler now groups metadata by the declared attention window. Each group
has an explicit initialization and producer dependency with a serialized `window`
attribute. The native validator checks producer/initializer and consumer windows.
The executor owns a stable metadata buffer per window, with a row per compiled
prefill token. Both vendor calls use `window - 1`; graph replay and the frozen
arena keep those buffers resident. Physical-plan descriptions expose this sharing
rule. Existing physical binaries without the attribute must be recompiled.

## Verification

- Local compiler token suite: 12 tests passed. The added test lowers a model with
  512- and 2048-token dense windows and verifies two groups, matching zero/producer
  attributes and correct consumer dependencies. This is a lowering test, not
  full-model NPU qualification of the 2048-window variant.
- Docker native plan validation: all eight initialization bindings passed;
  36 invalid contracts rejected, including missing or mismatched metadata windows.
- Selected learned-index C8 92B DSL, TP8/EP8, two residents, four-token prefill:
  discovery, preparation and frozen execution each matched all 99 standard
  fixture tokens with permuted shared pages and interleaved histories.
- vLLM Rust 0.25.1 HTTP checks passed fixture text, SSE/usage agreement, six
  concurrent requests, queue isolation, peer-prefill cancellation/replacement,
  context overflow and unsupported-sampling rejection, and graceful shutdown.
- Mixed workload: 504-token prompt TTFT **5198.081 ms**; short peer completed in
  **672.558 ms**. These are mixed-load measurements, not isolated decode TPOT.
- All eight ranks in native and HTTP audits: frozen arena, planned workspace,
  zero workspace growth, zero retained-step allocations/internal completion
  boundaries and zero activation host transfers.

Arena remains **12,451,957,248 bytes/rank**, workspace **130,916,352 bytes/rank**.
The HTTP pool returned to zero occupied bytes after 17 admissions.
See [native results](native-results.json), [HTTP results](http-results.json),
[native audit](native-audit.json) and [HTTP audit](http-audit.json).

## Provenance and remaining scope

Docker `inferfabric-92b` on `7.156.99.58`, working directory
`/data/p00603624/inferfabric/openpangu92`. Artifacts use prefix `window-metadata-`
with suffixes `source`, `discovery`, `workspace`, `preparation`, `bound`,
`qualified` and `http`. Rust was cross-compiled locally in WSL; native code and
NPU execution ran inside Docker. The existing Rust serving binary was unchanged.

| Artifact | SHA256 |
|---|---|
| execution.bin | ae7c97fe4473488e0fd4829401d1d043ff3b2037ce1b71624aaab50c50576939 |
| native library | da34f18deea69dae4024a712043a49dd60c456939a5125b67f3a3146f6361790 |

The [ring kernel contract](../2026-10-17-sliding-window/README.md) is separately
qualified. Production ring allocation, physical-page ownership versus absolute
table expansion, manifest retention semantics and actual scatter-wrap regression
are not yet implemented. The state pool currently treats capacity as the maximum
request length; ring storage needs an explicit distinction between retained
capacity and executable context. Prefix snapshots must also preserve retained
state before ring writes reuse pages. Existing reference boundary divergences
remain unresolved; passing short standard fixtures does not qualify 64K output.
