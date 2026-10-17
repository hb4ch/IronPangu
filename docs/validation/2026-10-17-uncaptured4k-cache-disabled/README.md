# Uncaptured 4K capacity: cache-disabled regression

The full 92B TP8/EP8 process completed successfully with `ACLNN_CACHE_LIMIT=0`.
All 123 generated tokens across 12 requests matched the independent fixtures,
including forced continuations. See [the measurements and artifact hashes](summary.json).

This diagnostic run follows the [teardown failure](../2026-10-17-uncaptured4k-discovery/README.md).
Disabling the implicit vendor operator cache avoided that failure in this run;
it does not establish the underlying corruption's cause. Explicit prepared
invocations and decode ACLGraph remain enabled.

Each rank recorded only the decode graph, with 112 replays, and 14 uncaptured
prefill forwards covering 9,902 query tokens. These include two 4,096-token
zero-input warmups. Actual reference prompts were 12, 15, 14 and 529 tokens,
each repeated three times. No real 4K or 64K prompt is qualified by this test.

The 529-token prompt's TTFT was 25.27, 20.39 and 9.74 seconds; TPOT was
24.40, 23.98 and 24.57 ms respectively. Total process time was 528.24 seconds,
including initialization and preparation. These are individual measurements,
not a steady-state throughput benchmark. Performance remains a release blocker.

This was allocation discovery, not frozen-arena serving. The next validation
must use a fresh discovery for the batched MoME binary, plan shared workspace
before binding, then verify the bound arena and long-context reference outputs.
