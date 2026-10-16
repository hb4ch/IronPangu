# Refreshed reference and timeline — 2026-10-16

The repeated OmniInfer TP8/EP8 benchmark is **18.1817 ms/token** over 60
intervals, versus 18.1171 earlier. All 66 greedy outputs match the same pinned
cases. It uses the same Docker/checkpoint/devices, context 512, batch one,
prefix cache disabled, greedy decoding and five diagnostic log probabilities.
The selected native path remains 21.1723 ms/token, **16.4% slower**.

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
