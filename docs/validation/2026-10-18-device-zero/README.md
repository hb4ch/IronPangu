# Planned device zero — 2026-10-18

The compiler now selects `aclnnInplaceZero` explicitly for the 88 physical
`expert_zero` instructions. The executor follows that selection using an INT8
byte alias over contiguous BF16 output storage: all-zero bits represent BF16
+0, without conversion or activation readback. The old memset selection remains
executable for existing plans. No implicit runtime choice overrides the plan.
Recipes and the hierarchical viewer declare the selected API and byte alias;
vendor-internal ZerosLike/ViewCopy behavior is not claimed to be one kernel.

Local verification: 11 compiler unit tests and 7 precision/physical-plan tests
passed, including the emitted kernel and viewer recipe. The real generated
viewer passed `scripts/test-token-viewer.cjs`. Rust was cross-compiled in local
WSL; C++ was built inside the NPU Docker container.

Full 92B qualification used the existing independent 4088-token reference.
Discovery, fixed-workspace and frozen stages each matched five tokens through
EOS, with decode graph replay. Frozen execution passed all eight ranks' zero
activation-transfer, zero internal-boundary and zero execution-allocation gates.
Arena remains 39,572,690,960 bytes/rank; workspace remains 159,221,248 bytes/rank.

Prefill ABI timings in seconds:

| Stage | Previous lazy-row implementation | Device zero |
| --- | ---: | ---: |
| Discovery | 7.417684 | 3.443793 |
| Fixed workspace | 6.817797 | 3.782884 |
| Frozen arena | 6.729598 | 3.172726 |

These are single samples per stage, not a statistical benchmark or production
SLO. The frozen sample is about 53% lower. Initialization is excluded from these
ABI measurements. Token request elapsed times include admission/reset work and
are a different measure. Long-context HTTP qualification is a separate gate.

Remote artifacts: `device-zero64k-source`, `native-build-device-zero`, and
`device-zero4k-frozen-plan`, under `/data/p00603624/inferfabric/openpangu92`.
Three-stage runner completed successfully (HCCL port 63000). Existing lazy-row
and older backend/plan pairs were preserved; fingerprints were freshly derived,
not retagged. This change addresses repeated expert-output clearing only.
