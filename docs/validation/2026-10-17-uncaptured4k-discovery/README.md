# Uncaptured 4K capacity discovery — failed teardown

This is **not a passing qualification**. See [the recorded evidence](summary.json).

The full native 92B TP8/EP8 engine performed two 4,096-query zero-input preparation
forwards without prefill graph capture. It then generated all 123 expected tokens
for four independent-reference fixtures, repeated three times. Prompt lengths
were 12, 15, 14 and 529 tokens; all generated tokens matched, including the forced
continuations. This does not establish numerical correctness for a real 4K or
64K prompt. Only the decode graph appears in all eight rank audits; activation
host transfers are zero.

The process subsequently failed during teardown with `free(): invalid pointer`
and exit status `-11`. The run took 850.06 seconds including initialization,
preparation, inference and failure handling. A system core was preserved and
examined inside Docker. The available crash frames resolve to
`libnnopbase.so: op::internal::OpExecCache::OldCacheClear()` and `Finalize()`.
The core is limited to 2 GiB and does not provide a complete caller stack; these
frames locate the failure but do not establish its root cause.

Each rank discovered 38,253,633,024 bytes of arena slots and a peak shared-workspace
request of 159,221,248 bytes. This was discovery, not a frozen-arena run: neither
no-growth serving nor variable-tail allocation completeness has been qualified.
The 529-token fixture's TTFT ranged from 39.56 to 51.63 seconds. Its poor speed is
a release blocker. Each rank recorded 1,366,890 preparations of the row-wise MoME
operator over warmup and requests; this count motivates native sequence batching
but is not a measured attribution of wall-clock time.

A standalone router executor creation/destruction probe at widths 16 and 4,096
passed on NPU. It did not reproduce the full-model teardown failure.

The next diagnostic run disables the vendor operator cache with
`ACLNN_CACHE_LIMIT=0`, which is supported by the
[CANN 9.1 environment-variable documentation](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/maintenref/envvar/envref_07_0031.html).
That is an isolated diagnostic configuration, not yet a validated serving fix.
Decode ACLGraph, device residency and complete output comparisons remain required.
