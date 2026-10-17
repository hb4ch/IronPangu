# Completion-scoped prefill invocation qualification

The full 92B TP8/EP8 process exited normally. All **159 normal-EOS tokens across
21 requests** matched the independent references. Seven prompt lengths
(12, 15, 14, 529, 2049, 3072 and 4088) were each repeated three times; sampled
tokens, logits and top-five diagnostics were exactly repeatable. Forced outputs
after the first EOS are outside this qualification scope.

See [measurements, artifact hashes and all-rank audits](summary.json).

Uncaptured prefill now caches repeatable ACLNN invocations only until the existing
owning-stream completion boundary. Decode/graph invocations use a separate
permanent cache. Reclamation adds no stream or device fence. The native probe
alternates transient and permanent invocations before completion to check reuse,
outputs and separation; it passed on NPU. The previously qualified distinct MoME
scratch allocation is included in this binary.

All eight ranks reported zero remaining transient invocations and zero activation
host transfers. Rank 0 prepared 3,456,622 transient invocations over the run, with
a peak of 377,008 retained until completion and 486 permanent invocations. These
counts expose substantial remaining host overhead; bounded lifetime does not
prove a low or bounded process RSS. An observed late-run RSS sample was about
194 GiB (not a sampled peak or a production memory guarantee).

Each rank retained only the decode graph (139 replays). The 23 uncaptured forwards
and 37,529 query tokens comprise two 4096-token preparation forwards plus 21
actual request forwards / 29,337 prompt tokens. No prefill graph was used. Arena
discovery was 38,253,633,536 bytes and maximum workspace 159,221,248 bytes per rank.
This run did **not** use a frozen arena; fixed-workspace preparation and frozen
qualification are separate subsequent gates using the same binary.

| Prompt tokens | Engine TTFT across three runs | Mean TPOT per run |
| --- | --- | --- |
| 2049 | 40.86–50.15 s | 146.88–205.32 ms |
| 3072 | 69.04–73.57 s | 209.57–359.34 ms |
| 4088 | 107.50–114.81 s | 318.18–391.84 ms |

These are native engine measurements, including the current completion and
sampling work; they are not HTTP latencies or isolated kernel timings. Total run
time was 1221.70 seconds including loading and preparation. Performance remains
inadequate. Per-row projection work and the large invocation count still need
optimization; this milestone establishes correctness and reclamation semantics,
not production performance.

Local compiler contract tests, the Docker native build and NPU prepared-execution
probe passed. The five Python qualification tests also pass, including explicit
separation of preparation/request forward counts and one-token tail handling.
The discovery process loaded the earlier report schema; subsequent reports also
expose these counts as structured fields without changing native execution.

No multi-chunk continuation, 64K, HTTP, mixed scheduling or APC qualification is
claimed by this artifact.
