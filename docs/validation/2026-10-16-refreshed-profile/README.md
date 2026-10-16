# Refreshed reference and timeline — 2026-10-16

The repeated OmniInfer TP8/EP8 benchmark is **18.1817 ms/token** over 60
intervals, versus 18.1171 earlier. All 66 greedy outputs match the same pinned
cases. It uses the same Docker/checkpoint/devices, context 512, batch one,
prefix cache disabled, greedy decoding and five diagnostic log probabilities.
At that milestone the native path measured 21.1723 ms/token. The subsequent
[paired-projection milestone](../2026-10-16-paired-projections/README.md)
reduced it to **20.0132 ms/token**, still **10.1% slower** than this reference.

Native profiling now defaults to task timelines without hardware performance
counters; set `INFERFABRIC_PROFILE_METRICS=1` to collect counters explicitly.
The options are saved beside each capture. Full Chinese output (17 complete
records) still matches the qualified fixture, with no activation host transfer.

Removing counters did not explain the mHC duration difference: native mHC pre
still sums to about 2.86 ms/token for 93 calls, versus about 1.55 ms for 92 in
the reference trace. Both use eight cube and sixteen vector blocks and FP32 phi
weights. The earlier V2 API experiment did not improve unprofiled TPOT.
The cause is not established. Native BF16 dense matmuls (178 calls, about
2.17 ms/token) are another candidate for core-allocation experiments.

Inclusive kernel sums are not additive TPOT; kernels and communication overlap,
profiling perturbs execution, and the reference trace used a different prompt.
Unprofiled wall times decide whether an optimization is retained. The current
profile is diagnostic evidence, not a performance parity claim.

An eight-cube/eight-vector scope around one-row BF16 projections with both
weight dimensions <=2560 passed full output and replay audits, but measured
21.3748 ms/token. It was reverted; smaller core allocations did not improve
this workload. The experiment summary is retained alongside this report.

Follow-up experiments on the paired-projection plan retained the exact complete
output fixture and passed all 48 rank residency/replay audits per experiment:

- Staging mHC phi through a device multiply by one measured 19.9591 ms/token
  (median 19.7563). The mean improvement was only 0.27%, with an outlier, so this
  was not selected as an established improvement.
- Capturing output top-k, log-softmax and rank consensus with the decoder
  measured 20.0369 ms/token. It did not improve the existing output boundary.
- Using NCL metadata for the mHC residual, matching the reference input format,
  also failed to improve TPOT; see its retained summary for exact timings.

All three experiments were reverted. Their summaries identify the measured
binary, execution plan and qualification fixture. The selected implementation
remains the paired-projection milestone; none establishes performance parity.
