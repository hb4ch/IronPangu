# Collective domains and decode graph coexistence — 2026-10-18

The standalone `collective_domain_probe` passes for both HOST and AI_CPU bulk
communicators, while a distinct HOST communicator remains available for decode
ACLGraph. It uses eight ranks, 10,485,760 FP32 elements (40 MiB per rank), three
warmups and 20 measured reductions. Every synthetic output element is checked.
Three HOST graph replays are interleaved with further bulk reductions; both paths
are checked after each alternation. No device reset or model execution is used.

| Bulk mode | Median slowest-rank completed call | Range | Median rank enqueue time |
| --- | ---: | ---: | ---: |
| HOST | 35.983 ms | 35.151–36.650 ms | 34.792 ms |
| AI_CPU | 0.899 ms | 0.837–1.898 ms | 0.304 ms |

Completed calls include the owning stream's completion. Enqueue time ends when
HcclAllReduce returns. Ranks are grouped by loop iteration; these are synthetic
collective measurements, not model TTFT/TPOT. Constant integer-valued FP32 data
checks transport/reduction correctness but does not qualify model rounding or
quality. No full-model speedup or production default change is claimed.

This supports a targeted next step: declare separate communicator domains in the
physical plan, with AI_CPU for uncaptured bulk AllReduce and HOST for decode and
other collectives. Keep mode selection, buffers and per-variant use inspectable;
initialize both domains before preparation. Preserve reduction dtype, rank order,
state residency and uncaptured prefill. Qualify the full 92B output and frozen
resource contract before HTTP/64K and latency comparisons. Do not dynamically
mutate a process-global environment variable around calls or switch modes in a
captured graph.

## Reproduction

Build the permanent diagnostic target inside Docker `inferfabric-92b`:
`cmake --build native-build-output-rows --target collective_domain_probe -j4`.
With the normal CANN environment, global `HCCL_OP_EXPANSION_MODE=HOST` and
120-second HCCL connection/execution timeouts, run:

```
HCCL_IF_BASE_PORT=43400 native-build-output-rows/collective_domain_probe 1 8 10485760 20
HCCL_IF_BASE_PORT=43600 native-build-output-rows/collective_domain_probe 2 8 10485760 20
```

The mode is explicitly assigned in HcclCommConfig before communicator creation;
the global HOST setting remains fixed. The installed CANN header defines 1=HOST,
2=AI_CPU, 3=AIV; no newer source-only enum is assumed. Initialization bootstraps
device zero before acquiring root info. Both final runs exit 0.

Workspace: `/data/p00603624/inferfabric/openpangu92`. Probe SHA256:
`712eee55968751a9cf7ba6ef6a54eaeaeeda75ebdf5d56eb8ddf56725a548ef5`.
Raw per-rank/per-iteration timings, logs and the pre-run read-only memory query
are in `collective-domains-v2/`; `summary.json` aggregates those raw samples.
