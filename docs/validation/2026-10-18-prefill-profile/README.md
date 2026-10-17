# One serving prefill capture — 2026-10-18

The persistent native serving ABI completed the independently qualified
`access-code-4088` fixture through its first EOS: five exact tokens. The first
batch processed all 4088 input tokens in one uncaptured forward. Subsequent
decode used the existing graph path. This is a diagnostic capture, not a
performance improvement or production latency qualification.

The probe-only profiler window excludes initialization and includes exactly one
batch, as recorded in `tokens.jsonl`. Its ABI duration was 6,808,319 us with
profiling enabled. Request elapsed time additionally includes profiler setup and
export teardown and must not be treated as normal TTFT. All eight ranks retained
the 39,572,690,960-byte arena and 159,221,248-byte workspace, with no forbidden
activation transfer, execution allocation or internal completion boundary.

The execution library and frozen `pooled8k-clean-frozen-plan` were unchanged.
Only `program_engine_probe` was rebuilt inside `inferfabric-92b`. The fixture
used existing `sequence8k-fixture/page-bindings.txt`; the qualifier ran with
`--frozen-only --batch-trace`. HCCL port was 62000.

`kernels.json` summarizes the exported AI_CORE tables for one iteration. It
does not cover all device task kinds, communication, host submission or idle
intervals. Do not interpret its sums as a critical path. Follow-up must inspect
those omitted parts before selecting an optimization.

Raw CANN capture and exported timelines remain in the remote Docker workspace:
`/data/p00603624/inferfabric/openpangu92/prefill4k-profile`.
Native qualification output is `prefill4k-profile-check-frozen` beside it.
`msprof --export=on --output=prefill4k-profile` completed successfully; cluster
tuning reported incomplete, so no cluster-tuning conclusion is claimed.

Before the run, the authorized `test_npu_load` pressure container was stopped;
a read-only ACL memory query confirmed approximately 65 GB available per device.
No device reset was performed.
