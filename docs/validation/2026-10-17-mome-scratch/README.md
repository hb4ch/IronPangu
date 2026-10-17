# Sequence MoME scratch fix: normal-EOS qualification

The full 92B TP8/EP8 run completed normally and matched all **114 normal-EOS
output tokens across 12 requests**. The independent repeatability checker passed
for all four fixtures repeated three times: sampled tokens, logits and top-five
diagnostics were identical. All 114 diagnostic samples also exactly matched the
earlier row-wise MoME baseline (`uncaptured4k-v2-discovery`).

See [measurements, hashes and all-rank audits](summary.json). This validates the
fix for the [unsafe scratch alias](../2026-10-17-mome-alias-investigation/README.md):
batched MoME now receives distinct BF16 `yOut` storage. The vendor kernel writes
scratch, synchronizes its cores, then writes back to the input. Scratch uses an
ordinary device-pool lease and appears explicitly in the physical recipe.
No extra host transfer, host fence or prefill graph was introduced.

The run retained the committed executor-cache behavior to isolate this fix.
Completion-scoped executor reclamation is a separate, still-unqualified candidate.

Each rank recorded only the decode graph (103 replays), zero activation host
transfers, and 14 uncaptured forwards / 9,902 query tokens including two zero-input
4K warmups. Arena discovery was 38,253,633,536 bytes and maximum workspace was
159,221,248 bytes per rank. These slots were not yet frozen.

The 529-token fixture's TTFT was 30.05, 22.13 and 7.21 seconds; TPOT was 24.85,
24.74 and 24.68 ms. Total run time was 482.13 seconds including initialization
and preparation. Performance remains inadequate; this is a correctness fix.

Scope: actual prompts 12/15/14/529 tokens; normal first-EOS stopping. No 4K/64K
prompt, frozen-arena, chunk-boundary or HTTP/APC qualification is claimed here.
The compiler scratch-shape/non-alias contract test and Docker native build passed.
