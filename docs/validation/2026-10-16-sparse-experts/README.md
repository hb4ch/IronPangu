# Sparse owner-local grouped experts — 2026-10-16

The backend now executes only the selected experts owned by each rank. Routing
IDs, owner masks, ordering and group counts remain on device. The DSL still
specifies the model and expanded 2,109-node execution program.

Each rank forms a device sparse group list, with local routes first in original
route order and zero-count foreign entries at the end. CANN grouped INT8 matmul
reads the owner-local weight bank directly and stops at the first zero count.
Gate, up and down each use one grouped call rather than eight selected-weight
copies and matmul calls. Ranks with no selected expert return explicitly zeroed
rows. Expert reduction retains local route order in FP32, followed by the existing
FP32 HCCL all-reduce. This is replicated-token EP with owner-local computation,
not an all-to-all dispatch path for independently batched requests.

Checkpoint scale storage must be BF16; the runtime checks source dtype from the
initialization plan before using the grouped kernel. Repacking BF16-origin scales
from initialized FP32 is lossless. Other scale contracts fail explicitly; there
is no implicit host fallback. Additional packed-bank memory remains an open
initialization-planning optimization.

## Numerical qualification

Grouped dequantization is a distinct numerical implementation. It preserves the
complete greedy output on the three pinned reference prompts, but does **not**
preserve the previous native logits bitwise. The original fixtures remain intact.
A one-row quantization experiment did not restore bitwise agreement, so changing
batch quantization alone was not sufficient to explain the numerical difference.
A separate mHC V2 experiment was reverted; this milestone retains mHC V1.

All 33 eager output tokens, including EOS, match pinned OmniInfer. Maximum
selected-token log-probability deltas versus that reference are 0.091099,
0.007573 and 0.229848 for arithmetic, geography and Chinese. The old path's deltas
were 0.128368, 0.030450 and 0.350542. Versus the old native path, maximum selected
logit deltas are 4, 4 and 5; log-probability deltas are 0.037269, 0.030453 and
0.498607. This is limited greedy-output qualification, not broad distribution or
production-quality qualification.

The graph benchmark must match every output record of this independently
reference-checked eager implementation exactly. It also checks all rank transfer
ledgers, zero generated-step pool allocation/completion, bounded request-output
transfers, and graph replay. No internal or device-global fence was introduced.
Physical-plan candidate metadata exposes the grouped numerical contract and its
BF16 scale requirement; the generic viewer displays these without model labels.

Fresh-process HCCL initialization intermittently reported port reuse during
benchmarking. The harness now optionally assigns disjoint 32-port ranges per run
and records them, without changing per-token execution. Failed partial runs are
excluded from reported performance.

## Measured milestone

**41.3654 ms/token**, 24.1748 tokens/s; median 41.2715 ms, empirical p95 42.3792 ms.
Same TP8/EP8, devices 0–7, checkpoint, batch one, context 512 and three prompts
repeated twice as the matched reference. All 66 graph output records match the
new eager records exactly, with 48 passing rank audits and 60 timed intervals.
Latency is 35.0% lower than the BF16/view milestone (63.6728 ms), and 45.6% lower
than the first native graph (76.0206 ms). OmniInfer full graph is still faster at
18.1171 ms: **2.28× gap remains**. See the
[matched methodology](../2026-10-16-native-graph/README.md); these are short-context,
batch-one offline measurements, not a production throughput/SLO claim.

All 42 compiler tests passed locally, including explicit sparse-kernel numerical
contract/viewer metadata checks. The compiler was cross-built locally; native C++
and all NPU runs remained inside Docker. The regenerated DSL bundle keeps the
same execution.bin hash. The six-run benchmark used `--decode-graph
--resident-audit --repeats 2 --hccl-base-port 32000`, with `eager-results.json`
from this milestone as the exact replay fixture after independent reference
comparison. Raw interval/output records, binary/fixture/program hashes, and
numerical comparisons are checked in alongside this document.

## Follow-up graph trace

`device-profile.json` captures two warmed graph decode iterations of this milestone
on all eight devices and preserves hashes of the parsed profiler databases. The
full 17-token output passed its exact-record and residency checks during capture.
On rank 0, mHC pre accounts for approximately 9.69 ms/token, sparse grouped matmul
2.30 ms/token, and attention 2.36 ms/token. Thousands of remaining elementwise and
copy tasks motivate the next vector-kernel milestone. Profiling overhead and
cross-stream overlap preclude treating summed task time as benchmark TPOT.
