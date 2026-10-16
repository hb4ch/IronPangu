# Local physical-planning and device-contract verification

Validated locally in WSL on 2026-10-16. No NPU was occupied or native inference
rerun for these milestones. The qualified token execution program remains
byte-identical (SHA-256
`46fa67de15080a6ccf370ea852f2d7a6744b532603bdf73ded7f59c34a082513`).

Milestone 1 (`cbc337d`) adds the design, physical inspection IR, conservative
lowering and viewer. Milestone 2's contract work adds checked device allocation
and tensor-view binding to the existing native kernel ABI. The actual physical
executor and migration of native kernels remain subsequent work.

## Generated plan

[Summary](summary.json) records eight rank/device bindings, one shared serial
schedule template, 2,109 calls and 321 collectives per rank. The partial known host
output-payload peak is 681,984 bytes. This is neither total host memory nor HBM.
Workspace/internal state/device totals and latency estimates remain unknown.

The physical plan is explicitly non-executable, with no arena reuse or overlap.
It exposes the host-staged backend's opaque call boundaries rather than inventing
individual device launches. The native executor continues using `execution.bin`.

[Open the viewer](../2026-10-16-reusable-dsl/execution.html), select **Physical
plan**, choose a rank, then inspect call order or activation storage. Selecting
a row shows implementation choices, costs/provenance, lifetimes, collective
signature and unresolved requirements. The device-buffer support panel describes
the available ABI contract without claiming a bound invocation already exists.

## Checks

- Workspace Rust tests and Clippy with warnings denied passed.
- Physical lowering is deterministic; costs distinguish unknown from numeric
  estimates. Synthetic candidate-cost tests verify legality, conservative cost
  selection, deterministic ties and missing-cost fallback. They are not hardware
  performance measurements.
- Corrupt rank/device bindings, collective participants/order, dependencies,
  selection indices, memory summaries, lifetimes and unsupported reuse/executable
  claims are rejected. The renderer checks logical-plan identity/mapping.
- Device contract tests cover disjoint subviews, input read/read overlap,
  serialization and 64-bit POD ABI offsets. Negative cases cover writable aliases,
  unknown IDs, placement, bounds, overflow, unsupported strides, workspace growth,
  readonly resources and duplicate/overlapping resident allocations.
- Binding tests use host-only mock allocations/addresses and never enqueue NPU
  work. They cannot establish ACL allocation provenance or asynchronous completion.
- Browser checks visit 460 scopes and 462 logical pages, compare all displayed
  dependencies, exercise physical rank/page/storage controls, and verify selected
  decisions and unknown costs. No view exceeds 24 cards; no JavaScript errors or
  HTTP requests occur. Physical-view screenshots were visually inspected.

## Reproduce

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p inferfabric-compiler --example view_token_plan -- \
  docs/validation/2026-10-16-reusable-dsl/execution.json execution.html physical.json
node scripts/test-plan-viewer.cjs execution.html
```

The browser check uses Playwright as described in [the viewer guide](../../plan-viewer.md).
`physical.json` is also emitted by `compile-token` inside new bundles. Its hash
depends on the adapter/backend contract sources; unrelated file timestamps do not
enter the artifact. See [the accepted design](../../native-physical-planning.md)
for the remaining execution, memory-reuse, profiling and optimization milestones.
