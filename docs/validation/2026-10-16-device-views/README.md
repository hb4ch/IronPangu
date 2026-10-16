# BF16 storage and shared device views — 2026-10-16

Native full-graph TPOT is **63.6728 ms/token** (15.7053 tokens/s), versus
76.0206 ms before this change: 16.2% less latency. The matched OmniInfer full-graph
reference remains 18.1171 ms/token, so the remaining gap is **3.51×**.

Same `/models/92B`, Docker, devices 0–7, TP8/EP8, batch one, context 512, greedy
max-32 generation, three prompts repeated twice. All 66 complete output records
match the pinned reference exactly. All 48 case/rank audits passed: no activation
host transfers, no generated-step allocations or internal completion boundaries,
expected bounded output bytes, and graph replay on every rank. The benchmark
measures 60 intervals after first output; loading/prefill are excluded. See
[the reference methodology](../2026-10-16-native-graph/README.md) for comparison
limits. This is not a concurrent or long-context serving result.

## Changes

- BF16 operator results retain BF16 storage instead of copying to an intermediate
  allocation and casting to FP32. Arithmetic, reductions, routing and diagnostics
  retain their explicit FP32 boundaries; numerical rounding is unchanged.
- Device results share ownership of their pool allocation. Contiguous slices
  retain that same lease and change only offset/descriptor. The final alias
  releases the block for ordered reuse on its owning stream.
- Matmul, quantization, normalization and collectives read compatible device
  operands directly instead of copying into temporary input tensors.
- Physical metadata and the generic viewer report actual BF16/FP32 payloads and
  shared-lease slices. The model DSL and execution binary are unchanged. Static
  physical arena placement remains unresolved; payload sizes are not total HBM.

The asynchronous prepared-execution probe now checks nested-view offsets,
producer destruction, concurrent same-size allocations, last-alias release and
out-of-bounds rejection. Full 17-output eager and graph checks passed before the
six-run benchmark. All 42 compiler tests passed locally; physical tests additionally
check precision/payload metadata against the expanded 92B program. Rust was
cross-compiled locally and the regenerated bundle's execution binary matches.
C++ compilation and all NPU work ran inside Docker.

Evidence: `summary.json`, `runs.json`, `prepared-probe.txt`, and the eager/graph
correctness records. Reproduce with `benchmark-native-tpot.py --decode-graph
--resident-audit --repeats 2` and the same arguments as the previous milestone.

Remaining work: grouped device-driven expert execution, fused vector operations,
and faster mHC kernels. No claim of OmniInfer parity yet.
