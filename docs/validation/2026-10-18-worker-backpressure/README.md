# Full-model worker backpressure — 2026-10-18

Program serving now caps each request's generated-output queue at 64 messages,
independently of the compiled context. The existing scheduler pauses a resident
before further execution when its channel is full, allowing peers and
cancellation to proceed. This bounds this queue only, not every HTTP buffering
layer or the process's total memory.

The cross-built Rust NPU test directly fills a two-message test channel to reach
the identical worker branch before a short reference reaches EOS. It does not
force post-EOS generation or claim an HTTP socket test. The unthrottled baseline,
resumed slow request, concurrent peer and reused resident each matched the
independent eight-token reference through EOS. Cancellation retired a full
channel and released its resident. All five admissions' shared pages were
reclaimed, and shutdown left no active request ownership.

All eight ranks passed frozen-arena, fixed-workspace, device-residency and
no-internal-fence checks. Native decode graphs replayed. The test binary was
built in local WSL using `scripts/cross-frontend-tests.sh` and ran only inside
`inferfabric-92b` via `scripts/qualify-program-backpressure.py`. The normal local
test suite leaves this hardware test ignored; this explicit run passed.

Remote output: `/data/p00603624/inferfabric/openpangu92/eos-backpressure`.
The runner exited 0 (HCCL port 64000), with `device-zero-apc8k-frozen-plan` and
`native-build-device-zero/libinferfabric_program.so`. Results include hashes of
the test binary, backend, executable plan and reference fixture. Sustained HTTP
slow-client buffering and transport-disconnect behavior remain separate gates.
