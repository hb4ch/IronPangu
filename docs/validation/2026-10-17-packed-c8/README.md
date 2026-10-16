# Packed C8 qualification — 2026-10-17

Milestone 1 implements the OmniInfer-compatible attention-cache layout, DSA-only
DSL selection, fused tiled cache writer, direct packed attention, and explicit
device-only candidate metadata. It has passed full 92B TP8/EP8 generation for
arithmetic, geography and Chinese prompts at context capacity 512. Complete
output token/logit records are in `c8-results.json`.

`kernel-checks.json` records the independent CPU checks: 512 cache rows, all four
per-128 scales and BF16 rotary values per row, plus 36,864 attention output values.
Attention was checked at positions 0, 31, 32 and 511, in eager execution and two
graph replay passes. Graph output matched eager exactly. Scale maximum absolute
error was 2.58313e-9, latent quantization differed by at most one integer at a
rounding boundary, and attention maximum absolute error was 0.00120384.
The all-zero row is included. Rotary storage matched the expected BF16 bits.

Native plan validation passed all eight initialization bindings and rejected
28 malformed contracts. Local Rust regressions check exact DSA layer selection,
explicit cache format attributes, one metadata producer, direct packed consumers,
and absence of full-cache reconstruction instructions.

Frozen-arena performance and memory qualification is in progress; these initial
results do not establish production quality, long-context DSA indexer parity, or
bitwise full-model parity with an independent OmniInfer C8 run.
