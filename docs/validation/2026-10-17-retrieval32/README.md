# 32K positional retrieval: correct through EOS, forced continuation differs

2026-10-17. Full native 92B TP8/EP8 execution with the prepared T=2/4/8/16
family, C8 learned-index attention, BF16 sliding-window state, two resident slots,
and a 32,768-token executable envelope. Three independent prompts contain
32,752 input tokens and reserve 16 outputs, with distinct answers near the start,
middle and end. Each ran three times with shuffled physical page mappings.

All **nine answers are correct**: early `8587`, middle `7863`, late `7998`.
All runs match the independent OmniInfer reference through EOS and are stable
across repetitions. All three full forced 16-token continuations differ after
EOS; the full-token comparison remains a failure. `comparison.json` retains
every native/reference output and the retrieval checks. No native tokens replace
the independent reference.

The native process completed normally in 1,086 seconds. Its wrapper returned
nonzero because full continuation parity failed. `process.json` distinguishes
that comparison failure from a native crash or timeout. Before the long run,
discovery, planned-workspace preparation and frozen replay each passed all 123
short/preparation fixture tokens, including a 529-token prompt.

All eight ranks retain a **13,281,872,896-byte frozen arena** and a
**131,020,288-byte shared workspace**, with zero workspace growth, activation
host transfers, rejected transfers, replay allocations or internal completion
fences. Each rank records 18,424 T=16 graph replays and 136 token-decode replays,
including warmup. The smaller variants have only their warm replay because these
long prompts are divisible by 16. The HBM sample shows 35,722–36,001 MB used on
native devices 0–7; this is a sample, not a measured lifetime peak.

The independent 64K reference ran concurrently on devices 8–15. Consequently,
the retained native timing fields are **not isolated performance evidence**.
This result does not establish production 32K robustness, 64K correctness,
HTTP tail latency, broad prompt quality or resolved numerical parity. Default
admission remains 512 and this run used the explicit experimental opt-in.

`identities.json` pins executable, backend, tokenizer, inputs, reference and raw
native output. Compact audits include hashes of their full remote source files.
`reference.json` is the independent reference; `model.inferfabric` is the exact
context-specialized DSL used for this run. Native builds and runs stayed in
Docker; the Rust compiler was cross-built locally in WSL.
