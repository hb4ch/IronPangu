# Repeated-request drift: MoME output scratch aliases input

Follow-up: the [isolated scratch fix passed normal-EOS and repeated-logit
qualification](../2026-10-17-mome-scratch/README.md). The failure evidence below
describes the earlier aliasing implementation.

The completion-scoped executor candidate failed normal-EOS output qualification:
the third Chinese request diverged at generated token index 5, before EOS. The
process exited normally in 485.92 seconds. Temporary executors were released and
activation host transfers were zero, but those resource checks do not establish
numerical correctness. See [the retained evidence](summary.json).

Comparison with earlier output diagnostics changes the investigation: the older
row-wise MoME run had identical Chinese logits across repetitions. The committed
sequence-MoME run already had drifting logits, even though its sampled tokens
happened to match. Executor cleanup is therefore not established as the source
of this drift.

Source inspection found an invalid alias in sequence MoME: the adapter passed the
same tensor as both `x` and `yOut`. The vendor implementation in
`ai_infra_fused_causal_conv1d_fn_cutbsd.h` writes intermediate outputs through
`CopyOut`, executes `SyncAll`, and only then runs `CopyOut2X` in inplace mode.
Aliasing the two buffers can overwrite input that another core still needs.
The scalar path already supplied a separate output scratch tensor.

The candidate fix supplies distinct `[actual_query_tokens, channels]` BF16 device
scratch, leases it from the existing arena pool, and exposes the scratch and
vendor writeback in the physical recipe. It adds no host transfer or fence.
The compiler contract test and complete Docker native build passed. Full-model
NPU qualification is pending; do not claim the race is resolved from source
inspection alone.

The redundant old-binary normal-EOS control was deliberately terminated after
the alias was identified; it is not a completed qualification. The new run uses
the committed executor-cache behavior to isolate the scratch fix. Completion-
scoped executor changes remain uncommitted until separately qualified.

Post-EOS differences are deferred per the user's scope decision. This failure
is before EOS and remains a correctness gate.
