# Surviving host allocation stacks — 2026-10-18

The resource gate still fails, now with allocation-site evidence. Eight mixed
load waves completed 40 independent-reference wave/recovery requests through
EOS. After each wave, the client awaited a completed allocator snapshot before
submitting the next wave. RSS grew from 7,597,363,200 to 7,944,957,952 bytes
(331.5 MiB), exceeding the unchanged 128 MiB limit. Threads and FDs were stable
at 632 and 78, including the diagnostic's thread and files.

The probe records one allocation per 1024 per thread, with exact removal of
recorded pointers on intercepted free and preservation on failed realloc. Across
the acknowledged first/last snapshots, unscaled sampled live bytes grew by:

| Allocation stack family | Sampled byte growth |
| --- | ---: |
| CANN `libnnopbase` | 313,240 |
| HCCL | 96,982 |
| Other | 6,265 |

These are sampled bytes, not exact total-heap attribution or proof of a leak.
The fixed table reported no dropped samples. Raw complete snapshots are in
`host-allocation-stacks.txt.gz`, with `stack-summary.json`, module aggregation,
and `symbols.json` containing installed DSO hashes and `addr2line` results.

The CANN stacks originate in `op::internal::BlockCache::CacheAllocImpl`, including
temporary Tensor construction for GMM, attention and contiguous/cast preparation.
The latter resolves through `aclOpExecutor::ConvertToTensor`, `l0op::Slice`,
`l0op::OptimizeContiguous`, `aclnnCastGetWorkspaceSize` and the native
`resident::column_slice` path. A libc allocation surviving does not distinguish
an object still in use from memory returned to CANN's internal pool. That is the
next ownership question to resolve before altering executor destruction.

HCCL stacks resolve through `std::deque<hccl::TaskInfo>::_M_push_back_aux`,
`TaskExceptionHandler::InsertTaskMap`, `Save` and `TaskRecordProfiler`. The local
HCCL source has bounded FIFO history (2048 tasks, or 65535 in op-base mode);
installed `InsertTaskMap` disassembly also shows the limit check and pop/free
path. This family must not be labelled an unbounded leak from short-run growth.
Source reference: `D:/cann/hcomm/src/legacy/ascend910/common/debug/profiling/task_exception_handler.cc`.

Normal SIGTERM drain completed this time: 75 admissions, 21 mixed ticks, 98,472
executed input tokens, zero final shared-page ownership and native workers
closed. All eight shutdown audits preserve arena 40,492,359,704 bytes and
workspace 159,221,248 bytes, with no workspace growth, execution allocation,
internal completion boundary, activation-host transfer or rejected transfer.
The overall runner exits 1 for the host resource gate; this is not a production
stability pass and no performance claim is made under allocation tracing.

Remote output: `/data/p00603624/inferfabric/openpangu92/host-idle-stack-http`,
HCCL port 65400. It uses the existing `device-zero-apc8k-frozen-plan`,
`native-build-device-zero` library and `inferfabric-server-apc-device-zero`, two
active residents, 4096-token uncaptured prefill, eight four-client waves, and
`--host-allocator-probe host_allocator_probe_idle.so`. Prefix caching is disabled.
The earlier periodic-only `host-stack-http` run also drained normally, but its
nearest-time samples can overlap prefill and are superseded by the acknowledged
between-wave snapshots above.

Container probe tests cover a known allocation/free, failed realloc, output
ownership, helper isolation, snapshot acknowledgement and frontend-style
`sigwait` shutdown. Twelve HTTP-helper tests and two allocation-summary tests pass.
The runtime library and frozen execution plan remain unchanged. Next distinguish
CANN pool-free blocks from live metadata and reproduce the dominant path with a
small operator loop; then apply and qualify the ownership/budget fix. Do not
raise the resource tolerance or discard decode graph metadata speculatively.
