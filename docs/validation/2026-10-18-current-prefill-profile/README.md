# Current 4K prefill profile — 2026-10-18

Profiled `dfe4d7f` runtime using `native-build-metadata-lease/program_engine_probe`
and `metadata-lease8k-frozen-plan`, without changing the runtime or arena. The
single capture starts after preparation and covers one 4096-token uncaptured
prefill at position zero. `profile_window` and `sequence_batch` records establish
the scope. It is not a mixed-batch or long-prefix capture.

The qualifier exits 0 with all 39 independent reference tokens across six
requests correct through EOS. All eight ranks preserve the frozen arena and
workspace, decode replay and device residency. This diagnostic run's profiled
ABI duration is 3,014,017 us; it is not an uninstrumented TTFT measurement.

## Observations and next decision

Rank 0 device task span is 3008.76 ms. Its 92 AllReduce intervals sum to
2561.36 ms, while notify-wait union is 2950.14 ms. Host API statistics across
all ranks contain 736 AllReduce calls and 111,296 notify-wait/record calls each.
These inclusive, overlapping intervals include dependencies and rank readiness;
they cannot be added together or interpreted as network transfer time.

For context, rank 0 MIX_AIC tasks sum to 238.27 ms and vector tasks to 85.10 ms.
The kernel summary's largest family remains sparse quantized attention (16 calls,
180.79 ms summed). This is not a critical-path attribution. The trace directs
investigation toward HCCL host expansion and dependency handling before changing
matmul tiling or shrinking the requested 4K quantum.

The earlier HOST/AIV environment comparison did not establish which implementation
actually ran and did not improve latency. Do not repeat it as an optimization.
Next inspect selected collective algorithms/expansion and API-to-device timelines,
separate host submission work from peer readiness, and test a targeted change on
this same reference fixture before another 64K qualification. Keep decode graphs,
uncaptured prefill, reduction semantics and the frozen memory contract intact.

## Reproduction

Docker `inferfabric-92b`, `/data/p00603624/inferfabric/openpangu92`, HCCL base
port 42200, normal CANN/vendor environment, HOST expansion and decode graphs on.
Set `INFERFABRIC_BATCH_PROFILE_DIR` to the new absolute `metadata-prefill-profile`
directory. Run `qualify-sequence-serving.py --frozen-only --batch-trace` with
the binary/plan above, checkpoint `/models/92B`, existing
`device-zero-apc8k-fixture` requests/bindings/reference, output prefix
`metadata-prefill-profile-check`.

Export with `msprof --export=on --output=metadata-prefill-profile`, then use
`summarize-serving-timeline.py` and `summarize-cann-kernels.py --iterations 1`.
Raw CANN capture/export remains in that remote Docker directory. Summaries record
source-table hashes; this directory retains complete qualification output and
export logs. Export success does not imply successful cluster tuning.
