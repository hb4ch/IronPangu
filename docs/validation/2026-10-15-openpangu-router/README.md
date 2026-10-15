# Real-weight OpenPangu 92B router qualification

Date: 2026-10-15. Roadmap OP-06 component milestone, not a full MoE layer or decoder.

The native router uses the real layer-2 gate matrix (BF16 checkpoint values cast to FP32), an FP32 matmul, and CANN `aclnnMoeGatingTopK`. It selects eight of 256 experts using sigmoid(logit) + correction bias. Combination weights use the unbiased sigmoid scores, normalize over selected experts with epsilon 1e-20, and multiply by 2.5. Group count and selected group count are both one. Expert IDs remain device-resident INT32 and weights remain FP32.

The operation parameters follow `omni/v1/models/pangu/pangu_v2_moe.py` and the installed CANN 9.1 headers. Local CANN documentation in `ops-transformer/moe/moe_gating_top_k/docs/aclnnMoeGatingTopK.md` confirms that sigmoid always normalizes even with renorm=0, and that correction bias affects selection rather than the output scores.

## Evidence

On 7.156.99.58 inside `inferfabric-92b`, device 0, the same image/CANN/driver as the [TP8/EP8 milestone](../2026-10-15-openpangu-tp8-ep8/README.md):

- Batches 1 and 4; three real-weight synthetic-input cases, including zero input.
- Four direct-logit edge cases: ties, bias overriding raw ordering, mixed positive/negative saturation, and all-negative saturation.
- Independent double-precision CPU dot-product and sigmoid/top-k/normalization checks. Selected expert IDs must be unique/in range and no excluded expert may beat the minimum selected corrected score beyond 2e-6. Ties permit equivalent selections, but replay must reproduce the exact eager IDs and scores.
- Three replay rounds per case for both complete router and selection-only graphs, with changed inputs/bias at stable addresses. Graph/eager IDs and scores match bit-for-bit.
- Batch 1: maximum projection absolute error 8.99658e-7, score error 3.80397e-8; 224 route entries checked.
- Batch 4: maximum projection absolute error 9.4017e-7, score error 4.13377e-8; 896 route entries checked.

Projection tolerance: 2e-5 * (1 + abs(reference)). Score tolerance: 2e-6 + 2e-5 * abs(reference). These are synthetic-activation component checks, not model quality/performance measurements. The standalone probe does not implement inactive-lane masking or dispatch/combine, and is not connected to the serving runtime yet.

The new floating-tensor exporter binds the complete main-model checkpoint before reading a named bounded BF16/F32 tensor, checks finite payloads, and records source/fixture hashes. Source checkpoint files are unchanged. Rust exporters were built/cross-compiled locally in WSL; C++ was built only inside the remote Docker container. All 68 existing workspace tests and strict workspace/all-target Clippy passed.

## Reproduce

After the [build setup](../../build.md), locally run `bash scripts/cross-openpangu-probe.sh`. Stage `.deploy/export_openpangu_tensor`, `native/`, and `scripts/qualify-openpangu-router.sh` into the development container, set the exporter executable, then run inside Docker:

```bash
bash scripts/qualify-openpangu-router.sh /models/92B router-qualification
```

See `reports/` and `fixtures/*.json` for results and identities. No weight payloads are committed.
