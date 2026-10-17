# Long prefill: uncaptured 4K forwards

Decision: 2026-10-17. This supersedes the small captured-prefill-family expansion
as the approach to long-context serving in `product-serving-plan.md`.

The amount of new work is `prompt_tokens - restored_prefix_tokens`, not the
absolute context length. Long new work runs as ordinary device kernel launches
in chunks of **4,096 tokens**, without ACLGraph capture or graph compilation.
Decode retains its prepared ACLGraph path. A 65,536-token cold prompt requires
**16 model forwards**; a 61,440-token prefix hit leaves one 4,096-token forward.
Each forward traverses the model once with a batched query dimension. Wrapping
4,096 one-token model traversals in one API call does not implement this contract.

## Execution and scheduling

- Compile the model's physical operator plan and a bounded prefill memory plan.
  Compiling this plan must not imply capturing a device execution graph.
- Declare the prefill envelope separately from exact captured decode shapes:
  maximum query tokens 4,096, execution `uncaptured`, variable valid query count.
  Its serving capability must say `graph_capable: false`. Runtime and viewer must
  agree about this decision. Never capture the large prefill during warmup.
- Select each chunk from the **remaining uncached suffix**. Launch up to 4,096
  actual query rows; run the final partial chunk with its actual valid length.
  Do not append padding into recurrent state or KV storage, or silently split
  tails into a sequence of small captured variants.
- Charge actual query tokens to `max_num_batched_tokens`. The normal long-context
  configuration must supply at least 4,096 tokens per prefill dispatch. Smaller
  explicit budgets may produce smaller forwards; they must not be reported as
  the 16-forward configuration. Mixed-workload scheduling must provide a full
  prefill quantum on a subsequent tick instead of repeatedly fragmenting it
  with the remainder after decode work.
- Restore APC state first. A long prompt with a short uncached suffix must not
  trigger the long cold-prefill policy merely because its absolute position is
  large. Keep positions, causal lengths and RoPE offsets absolute.
- Prefix publication follows completed forward boundaries that satisfy all
  state page alignments. Do not split a forward just to publish a checkpoint.
  For a cold 65,536-token prompt, the last reusable checkpoint before the final
  forward is 61,440 tokens. Retaining that checkpoint leaves one complete
  forward for a later identical prompt and avoids a seventeenth cold forward.
- Record the actual query count and uncaptured arithmetic class in continuation
  history. Do not infer numerical equivalence with a captured or differently
  chunked execution. A mismatch remains a cache miss.

## Device memory and kernels

Plan activation capacity, shared workspace and state retention before frozen
arena binding. Reserve storage for the maximum 4K chunk and provide bounded
views for tails. Exact-size allocation lookup for T16/T32 is not a variable-tail
allocation contract. Every allowed tail must fit the plan without runtime HBM
growth or accidental reuse of live decode-graph storage.

Keep activations on device. The input token IDs and declared control metadata
may be uploaded; output tokens may be returned. No implicit host activation
fallback, offload, global fence or runtime graph capture is allowed. Completion
remains on each owning stream at the request/chunk boundary.

Audit batched routing, grouped GEMMs, causal attention and index selection at
4,096 queries. Recurrent MoME state must preserve causal order within each
layer; this does not justify repeating the whole decoder per token. SWA storage
must retain the union of windows needed by all queries in a chunk while later
chunk rows are written. Avoid materializing full-vocabulary logits for every
prefill row when only the final prompt row is sampled; express that choice in
the physical plan rather than hiding it in the executor.

## Qualification gates

1. Scheduler tests: cold 65,536 -> 16 x 4,096; APC hit 61,440 -> one 4,096;
   non-aligned restored positions; a partial tail; exact budget accounting;
   mixed decode fairness; cancellation between chunks; no checkpoint-induced
   extra forward.
2. Compiler/runtime contract tests: prefill is uncaptured, decode stays graph
   backed, arbitrary valid tail views are bounded, and stale exact-shape-only
   backends reject the new envelope rather than silently falling back.
3. Native checks: full 92B outputs against the independent reference, chunk
   boundary and SWA-wrap cases, cold and APC-hit 64K requests. Count actual model
   forwards and prove that no prefill graph was captured or replayed.
4. Report TTFT, decode TPOT, peak HBM and per-rank arena/workspace audits. Preserve
   known forced post-EOS mismatches as failures; report through-EOS correctness
   separately. Do not lift the production context gate from scheduler tests.

## Implementation status

The superseded graph-prefill qualification and its dependent queued runs were
stopped on 2026-10-17. Their partial measurements are not a completed release
gate. The current executable still has exact small-prefill shapes and cannot
yet claim this 4K variable-tail contract. The next implementation milestone is
the bounded uncaptured prefill envelope, allocator and kernel integration—not
another captured-prefill shape family.
