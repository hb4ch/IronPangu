# Inference mHC and fused vector work — 2026-10-16

The sparse-expert graph trace identified mHC pre as the largest remaining kernel:
93 calls and about 9.69 ms per token on rank 0. Other measured costs included 1,376
scalar Mul kernels, 718 Add kernels, and 7,981 SDMA tasks. The
[raw aggregated graph trace](../2026-10-16-sparse-experts/device-profile.json)
contains all eight devices. Profiler overhead and overlapping streams mean task
sums are not TPOT; unprofiled benchmarks determine the latency result.

## Changes

- Select inference-only mHC outputs (`outFlag=0`). The installed vendor tiler's
  large-batch condition includes `outFlag == 1`, even for a single token. Removing
  the training-output request permits its small-token implementation. This uses
  the existing V1 API, not the earlier reverted V2 experiment. No training
  auxiliary tensor is consumed by the decoder.
- Fuse half-rotation, FP32 products and addition into RotaryPositionEmbedding,
  followed by the existing BF16 rounding. Position-derived FP32 constants remain
  in persistent device buffers. This change alone passed 17 complete output
  records exactly in both eager and graph mode against the sparse-expert fixture.
- Multiply the packed expert-row matrix by device scores and reduce it in FP32
  using vector operations, replacing eight serial row multiplications/additions.
  Parallel FP32 reduction is recorded as a distinct numerical contract.
- Preallocate multi-part concatenations and repetitions once, avoiding repeated
  copies of a growing prefix. Query/value heads and expert inputs use these
  generic device-buffer utilities. Allocation leases and stream ordering remain
  unchanged; no new host transfer or completion boundary is introduced.

Physical metadata and the generic viewer expose the inference mHC, fused rotary,
and vector weighted-reduction contracts. The logical DSL execution binary is
unchanged. Rust compiler tests and cross-compilation run locally; native build and
NPU execution run in Docker.

## Numerical qualification

All 33 greedy output tokens including EOS match the pinned OmniInfer reference.
Maximum selected-token log-probability errors are 0.032481, 0.004447 and 0.344681
for arithmetic, geography and Chinese. These results are not bitwise identical to
the old training-output mHC/serial-reduction path. Original fixtures are retained.
Broader prompt/distribution qualification remains open; this evidence establishes
complete generation on the three baseline prompts and exact eager/graph agreement
for the new implementation, not production-wide numerical equivalence.

## Measured result

**32.3856 ms/token**, 30.8779 tokens/s; median 32.2874 ms and empirical p95
33.3669 ms. All six runs passed 66 exact eager/graph output-record comparisons,
48 rank residency/replay/allocation/completion audits, and 60 timed intervals.
This is 21.7% less latency than sparse experts alone (41.3654 ms), and 57.4% less
than the initial native graph (76.0206 ms). The matched optimized OmniInfer
reference is 18.1171 ms, leaving a **1.79× gap**.

The same checkpoint, devices 0–7, TP8/EP8, batch one, context 512 and reference
prompts were used. Loading/prefill are excluded; this is a short-context offline
comparison. Graph records are checked against `eager-results.json` here after its
independent reference comparison. Original reference records are not overwritten.
All 42 compiler tests pass locally; the cross-built compiler produces the same
execution.bin and updated inspectable backend contracts. Raw runs and hashes are
in `summary.json` and `runs.json`.
