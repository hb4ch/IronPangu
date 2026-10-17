# Host allocator diagnostic — 2026-10-18

The 8-wave diagnostic narrows the earlier RSS regression: between the first
and last completed waves, RSS increased by 345,157,632 bytes and glibc allocated
arena bytes by 342,259,504 bytes. Free arena bytes increased by only 1,759,440
bytes; mmap bytes were unchanged. Nearest-time allocator samples are recorded
with their offsets in `allocation-summary.json`; sampling every two seconds is
not an exact quiescent heap census. This supports investigating retained live
allocations rather than treating the growth as only free libc arena retention.
It does not identify a leaking call site or distinguish an intentional growing
cache from a leak.

All 40 completed wave/recovery outputs matched their independent references
through EOS before shutdown. **This run is not a complete qualification pass:**
the diagnostic thread was created before the frontend blocked termination
signals for `sigwait`, so SIGTERM terminated the process without normal native
shutdown audits. No claim about this run's final page reclamation or rank audits
is made. Its remote output is `host-allocation-scoped-http`, HCCL port 65000,
runner exit 1. The earlier `host-allocation-http` attempt also had helper-process
preload inheritance interference and is not promoted.

The checked-in opt-in probe now restricts itself to the exact server executable,
ignores inherited vendor helper executions, avoids joining a parent's thread
after fork, and creates its sampling thread with signals blocked while restoring
the caller's original mask. Container tests pass known 64 MiB allocation/free,
existing-output rejection, helper isolation and a reproduction of the frontend's
post-startup `sigwait` shutdown. The final signal-safe probe has not yet undergone
a full NPU serving qualification; use that version for the next attribution run.

The installed CANN 9.1.0 library was also inspected before proposing ownership
changes. `aclDestroyTensorList` destroys its contained tensors; the ordinary
non-repeatable path in `CommonOpExecutorRun` destroys its executor. Adding extra
destruction to those paths would risk double-free. The native library, device
plan, descriptor lifetime and decode graphs were not changed by this milestone.

Next collect surviving allocation call sites across a short repeated workload,
then fix the responsible ownership/cache policy and re-run the existing 24-wave
resource gate. Do not raise its 128 MiB tolerance or repeat the 64K output ladder
before an actual runtime fix.
