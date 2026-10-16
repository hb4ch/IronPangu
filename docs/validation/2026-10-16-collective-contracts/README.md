# Explicit collective precision and AIV qualification — 2026-10-16

These experiments **do not improve the selected 21.1723 ms/token HOST path**.
The canonical 92B DSL retains FP32 collectives and the launch configuration
retains HOST. No performance parity claim is made.

| Configuration | Mean TPOT (60 intervals) | Complete graph records equal eager |
|---|---:|---|
| FP32 / HOST, prior milestone | 21.1723 ms | Yes |
| FP32 / AIV / deterministic=true | 21.7460 ms | Yes |
| BF16 / AIV / deterministic=true | 21.6962 ms | Yes |

OmniInfer defaults `hccl_op_expansion_mode=3` (AIV) when the environment does not
specify a mode. An initial native FP32 AIV run without deterministic reduction
kept the greedy sequence but failed exact complete-record comparison between
eager and graph. That run was rejected; the gate was not relaxed. Deterministic
AIV passed all six complete graph runs, but was slower than HOST.

The permanent generic addition is an optional `dtype` on `allreduce`: `f32`
(also the backward-compatible default) or `bf16`. It controls the input cast,
HCCL payload and returned device storage. Rust and native validation reject
unsupported values. The physical plan exposes storage bytes, collective dtype
and numerical contract. It does not silently trade precision for latency.
The [expanded experimental plan](bf16-execution.json) explicitly chooses BF16;
the canonical source does not.

BF16/AIV passed the full three-prompt NPU run with all 33 reference greedy
tokens. Selected log-probability differences are reported in the comparison
JSON; the largest is 0.365328 on the Chinese prompt. This is a distinct numerical
contract, not bitwise agreement with FP32 or broad quality qualification.
All 66 complete graph records equal its qualified eager fixture, and all 48
rank transfer/allocation/completion/replay audits pass. No activation readback,
implicit host fallback, per-step allocation, or global fence was introduced.

60 compiler/DSL tests pass locally. Native validation passes the experimental
and prior 1,699-node plans, all eight rank bindings and ten negative contracts.
Rust is locally cross-built; native builds and NPU runs are inside Docker.
Benchmark reports now retain complete output records even when a gate fails,
and record the expansion-mode and determinism environment settings.

These are batch-one, context-512, short-prompt offline decode measurements on
the same checkpoint and TP8/EP8 devices. Empirical p95 is not a production SLO.
