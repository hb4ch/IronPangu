# BF16 collective precision comparison

Date: 2026-10-17. Decision: **do not promote this experimental precision change**.
The selected learned-index DSL retains FP32 collectives. Context stays at 512;
full reference parity and 64K serving remain unqualified.

The experiment changes only the three reusable module call sites for attention,
dense FFN and routed-expert AllReduce to `dtype = "bf16"`. Expansion produces
92 BF16 collectives. The physical planner exposes input casts, HCCL dtype and
BF16 output storage. All other choices stay fixed: BF16 query rotary, INT8
indexer, C8 latent cache, TP8/EP8, HOST HCCL mode, two residents, four-token
prefill and shared shuffled pages. There is no native executor modification.

The exact source change is retained in `bf16-collectives.patch` against the
selected DSL at commit `e3c33e5`. Apply it in an experimental checkout and compile
with `--resident-sequences 2 --prefill-tokens 4 --shared-pages`; fresh workspace
planning and arena binding are required. The main DSL was restored after the
experiment, rather than selecting precision merely because the short fixtures
pass.

Results on `7.156.99.58`, inside `inferfabric-92b`:

- Standard full-model fixtures: **99/99 tokens pass** in discovery, preparation
  and frozen-arena execution.
- Boundary output: all **336 tokens** produced, identical across three
  repetitions. Four sequences differ from the prior FP32-collective native run.
- Original reference: **9/14** sequences match, unchanged in count.
- One-token-prefill reference: **9/14** match, versus **10/14** before. Failed
  lengths are 17, 31, 33, 64 and 65. Exact comparison correctly exits nonzero.
  Neither reference capture nor expected tokens were replaced.
- HTTP: full fixture text/SSE agreement, six concurrent requests, queue isolation,
  cancellation during peer prefill, replacement, context rejection, unsupported
  sampling rejection and graceful shutdown all pass. The shared pool reports
  zero occupied bytes after 17 admissions and 1,296 cross-partition bindings.
- Native and HTTP audits on all eight ranks show zero activation host transfers,
  workspace growth, replay allocations or internal completion fences. Frozen
  arenas and ACLGraph remain active.

Physical instructions: 2,760 (prior 2,768). Arena: **12,451,982,848 bytes/rank**,
25,600 bytes larger than the preceding FP32-collective arena. Shared workspace:
**130,916,352 bytes/rank**, unchanged. The 504-token mixed HTTP request has TTFT
**5.272 s**; the short peer finishes in **700.5 ms**. These individual observations
are not evidence of performance superiority. SSE text-chunk spacing is not TPOT.

Native validation checked all eight rank bindings and 33 negative contracts.
One negative test previously rewired AllReduce to the embedding and assumed it
must fail. A BF16 embedding with matching width is a legal BF16 operand. That
test now supplies a non-tensor value instead, preserving its intended rejection
independently of precision; no production validation was weakened. Local compiler
validation confirmed all 92 experimental collectives and their inputs/outputs
were BF16 before restoring the selected source.

Artifacts: `native-results.json`, `native-audit.json`, `boundaries-audit.json`,
`boundary-comparison.json`, `http-results.json`, `http-audit.json`, and the patch.

SHA-256:

- Experimental execution: `5348ee27ed55d30cb46cd08c279a587702501a23e6ed6a1653d0c37a1df36e27`
- Native library: `65188f8c091b827bc4e65ee13cfccc244765ad43f48bd87c6f5bf81be82f2175`
- Rust frontend: `f964e6e2c77df8806656ebc70ec7baba5a1673b2a3b3094e2a60760313746d06`

This rules out changing collective dtype alone as a demonstrated parity fix.
Further numerical work needs operator-boundary evidence rather than another
unattributed precision switch. Independent long-context state work remains
required: derive table capacities from planned state, distinguish absolute
positions from recycled window rows and retain bounded device allocation.
