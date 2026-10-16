# Maximum shared workspace planning

The preparation pipeline now separates discovery from arena construction. During
discovery, vendor `GetWorkspaceSize` and compiled-kernel preparation report their
requirements through the same `Context::workspace` interface. The audit records
maximum bytes per physical instruction and per rank, including preparation,
capture and ordinary token calls. Discovery may grow buffers but cannot bind an
arena.

`scripts/plan-shared-workspace.py SOURCE OUTPUT AUDIT_DIR...` merges maximum
requirements across the supplied runs. It writes `workspace.json` and one
`workspace-rank-N.txt` per rank. The executable format is:

```text
IFWORKSPACE1 PROGRAM_FINGERPRINT BACKEND_FINGERPRINT MAX_BYTES
```

Fingerprints bind the result to the execution program and native binary. The JSON
also records source-audit hashes and per-instruction demands. These are measured
bounds for the supplied workloads, not a proof about untested shapes or kernels.

A **fresh preparation run** consumes this plan. Before model/control allocation,
it acquires a single maximum-size lease and keeps it live for the context's entire
lifetime. Every vendor invocation and captured graph uses this address. Requests
larger than the plan fail before enqueue; there is no growth or fallback. Zero-byte
workspace plans allocate no slot.

The binder accepts only a planned, zero-growth audit with a matching sidecar,
consistent peak demand, a dedicated first slot of the exact planned size, and
instruction bindings covering every workspace user. The arena preserves the same
allocation order as fixed preparation. It does not delete slots from a historical
capture or relocate graph addresses. Fresh bound execution uses one arena
allocation and retains all addresses until its existing stream-local teardown.

This removes intermediate growth allocations without adding synchronization,
changing numerical kernels, or moving activations to the host. Sharing relies on
the existing single-stream contract; cross-stream use remains rejected. Future
concurrent-stream execution needs separate workspace lifetime planning.

The viewer exposes the shared slot, planned maximum, per-instruction request and
zero growth count from bound metadata. Tensor slots continue to use the existing
exact-size reuse policy. This change does not claim general interval packing or
optimal total activation memory.

## Validation

- `python3 scripts/test-shared-workspace.py`: merging multiple discovery runs,
  fingerprint and capture requirements, rejection of unplanned binding, overflow,
  growth and missing slot dependencies.
- `native-build/frozen_arena_test`: portable sidecar parsing and negative bounds,
  fingerprint and trailing-data contracts.
- `native-build/prepared_execution_probe`: fixed address across increasing demands,
  overflow rejection, normal buffer reuse and frozen-arena enforcement on NPU.
- Full 92B C8 and BF16 qualification: discovery, fresh fixed preparation, bound
  graph execution and C8 eager execution against unchanged output fixtures.

See [build procedure](build.md#frozen-arena-preparation-for-the-92b-token-backend)
and [measured qualification](validation/2026-10-17-shared-workspace/README.md).
