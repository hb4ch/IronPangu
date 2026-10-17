# Real 2K–4K prefill: comparison failure retained

The 92B TP8/EP8 native process exited successfully after 892.68 seconds, but the
qualification client failed its full output comparison. This is **not a passing
release gate**. See [outputs, timings, hashes and rank audits](summary.json).

Each of the 2049, 3072 and 4088-token prompts ran three times. The 2049 and 4088
fixtures matched all eight reference outputs every time. The 3072 fixture matched
the answer and EOS at index 4, but output index 7 was native `1114` versus reference
`148902` in every repetition. Forced post-EOS differences remain failures; normal
EOS agreement is reported separately and does not replace the complete comparison.

| Prompt tokens | Mean engine TTFT | Mean decode TPOT | Full comparison |
| --- | ---: | ---: | --- |
| 2049 | 39.45 s | 24.38 ms | Pass |
| 3072 | 80.87 s | 25.01 ms | Fail after EOS |
| 4088 | 56.23 s | 25.05 ms | Pass |

These timings include reset and prefill, not checkpoint loading or HTTP. The first
occurrence of a shape is included; these are not steady-state throughput numbers.

All eight ranks used the planned 159,221,248-byte workspace with zero workspace
growth. Only the decode graph was present (64 replays); 11 uncaptured forwards
processed 35,819 query tokens, including the two 4K zero-input warmups. Activation
host transfers were zero. The prepared arena was 38,103,138,816 bytes per rank.
It was not bound/frozen, so this does not establish allocation-free serving.

A process RSS snapshot near the end of output generation was about 282 GiB.
Code inspection identified permanent retention of repeatable executors for
uncaptured row addresses and shapes. A separate candidate uses single-use
executors in the uncaptured scope while preserving decode caching. Its full-model
qualification is pending; the observed RSS has not yet been attributed entirely
to that cache. Descriptor retention also remains a separate resource concern.

No 64K, chunk-boundary continuation, frozen-arena or HTTP qualification is claimed.
