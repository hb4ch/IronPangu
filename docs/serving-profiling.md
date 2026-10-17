# Profiling the native serving path

Profile the persistent serving ABI, not the legacy standalone decode client.
Use an existing frozen plan and independently qualified token fixture. Build
`program_engine_probe` inside the NPU Docker container; this does not require
changing the execution library or rebinding its frozen arena fingerprint.

Set `INFERFABRIC_BATCH_PROFILE_DIR` to a new, absolute directory and run
`scripts/qualify-sequence-serving.py --frozen-only --batch-trace` with the
qualified plan, requests, page bindings and reference output. Keep the normal
CANN/vendor environment and decode-graph settings used for qualification.

The probe starts the CANN profiler immediately before its first serving batch,
after weight loading, arena binding and engine warmup. It stops immediately
after that batch's existing synchronous ABI returns. It adds no device fence,
activation readback or graph capture. Later batches run without profiling;
the complete request still must match the reference through EOS.

A successful capture emits a `profile_window` JSON record with the captured
batch count, token count and sequence count. The corresponding `sequence_batch`
record includes the real member positions and token counts. Use these records
to establish the window; never infer the number of iterations from kernel count.
The trace duration excludes profiler start/stop but includes profiling overhead.
It is diagnostic timing, not an uninstrumented latency result.

Start with one roughly 4K prompt whose output is already independently verified.
Inspect the exported CANN timeline for host launch gaps, device compute and
collectives before changing kernels. `scripts/summarize-cann-kernels.py` can
summarize exported `ai_core_op_summary.db` tables with `--iterations 1` for this
window. Its summed kernel durations overlap across streams and cannot substitute
for wall-clock latency or a critical-path analysis.

After an optimization, verify the same small fixture without profiling, then
mixed prefill/decode scheduling and the necessary long-context regression.
Do not repeatedly run the full 64K suite while a smaller fixture reproduces
the issue. Keep the production execution library, frozen-plan fingerprints,
reference outputs and resource audits paired in each recorded milestone.
