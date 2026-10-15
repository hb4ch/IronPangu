# Complete OpenPangu 92B MoE layer: TP8 / EP8 correctness baseline

Date: 2026-10-15. Roadmap OP-06/OP-08 component milestone. This is one real-weight layer, not a complete 92B decoder or serving runtime.

Layer 2 loads all 256 routed experts, the FP32 router/correction bias, and the unquantized shared FFN. Eight ranks own 32 contiguous experts each. Shared gate/up/down projections use output-row TP sharding and canonical all-gathers, matching the placement plan's correctness-first layout. Every rank computes the same router and receives the same input.

The new Rust `export_openpangu_moe` example audits and binds the checkpoint once, then streams individual bounded tensors to 773 native fixtures. Each source payload and output fixture is hashed. Quantization scales must be positive and finite; floating payloads must be finite. It publishes the complete report atomically only after exporting all tensors. Checkpoint files are unchanged; binary fixture payloads are not committed.

## Arithmetic and execution

1. BF16 input is cast to FP32 for the learned router. Bias-corrected sigmoid top-8 selects experts; unbiased normalized scores are scaled by 2.5.
2. Device-side active-lane masking and scatter produce dense expert coefficients. No routing IDs are downloaded to control execution.
3. Each owner evaluates its 32 W8A8 experts. Gate/up outputs feed **fused `aclnnDequantSwigluQuant`**, then the down projection consumes the resulting INT8 values and FP32 per-token scales. There is no extra BF16 rounding between SwiGLU and requantization.
4. Each expert's BF16 output is converted to FP32, weighted, and accumulated. HCCL sums FP32 contributions across all eight ranks, including zero-route ranks.
5. The shared BF16 FFN uses native SwiGLU and three TP gathers. Its output is added exactly once to the routed FP32 sum, then rounded to BF16. Inactive lanes are zeroed in both branches.

This is deliberately **dense masked expert execution**: all resident experts compute, even with zero coefficients. It is a correctness baseline, not efficient dispatch. It also uses a declared FP32 combine policy; fused OmniInfer GMM/finalize variants can have different intermediate rounding. Full-model golden comparisons must qualify the chosen policy before any output-correctness or quality claim.

## NPU validation

Host `7.156.99.58`, Docker `inferfabric-92b`, devices 0..7, existing CANN 9.1 / driver 25.5.1 environment. Rust cross-built locally; all C++ builds and NPU execution inside Docker.

Batches 1 and 4 both passed four cases: two learned-input routes, bias forcing all selected experts onto rank 3, and inactive lanes. Each captured graph replays every case twice at stable addresses. Exact graph/eager output and expert IDs are required. Batch 1 checks 81,920 combined values and 24 active-expert evaluations; batch 4 checks 327,680 values and 73 active-expert evaluations. These counts are evaluations, not unique expert coverage.

Independent checks cover routing selection/normalization, scatter clearing and masking, active expert integer matmuls and fused activation quantization, shared projection CPU dot products, SwiGLU, every peer/token slice of all shared TP gathers, local weighted sums, inter-rank route consistency, FP32 all-reduction, final rounding, and zero-route ranks. Unselected experts execute but are not all independently CPU-checked. The earlier TP8 selected-expert probe still passes after the shared linear runner refactor.

Quantized activations permit one integer step; SwiGLU scale tolerance is 1e-6 + 2e-5 * expected. Projection/SwiGLU BF16 tolerance is 1e-4 + 0.016 * abs(reference). Combine tolerance is 1e-4 + 2e-5 * abs(reference). Graph outputs must match exactly. The output reports and per-fixture identities are attached here.

## Reproduce

Locally run `bash scripts/cross-openpangu-probe.sh`. Stage `.deploy/export_openpangu_moe`, native sources, and the qualification script; set the exporter executable. Inside Docker:

```bash
bash scripts/qualify-openpangu-moe.sh /models/92B moe-qualification
```

Remaining work: sparse dispatch/grouped expert execution, all-layer native loader/executor integration, mHC/MoME/MLA state, golden full-model logits and generation, and Rust frontend integration. Passing this layer test does not close those gates.
