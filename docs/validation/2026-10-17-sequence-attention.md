# Multi-sequence packed C8 attention prerequisite

The shared production `resident::packed_attention` function accepts a batch
page table, cumulative query ends and per-sequence KV lengths. The generic
sequence layout now produces these lengths, while `ResidentPages` constructs
tables in batch order without changing its selected resident. Ring tables use
each resident's own page cycle. Single-sequence cache table construction uses
the same implementation.

On 7.156.99.58 inside Docker `inferfabric-92b`, the production-helper NPU probe
`native-build-head-batch/sequence_attention_probe` returned:

```json
{"passed":true,"launches":3,"checked_values":12644352}
```

Each launch packs three sequences, with 3, 17 or 4096 total query rows. Two
single-row requests surround one extend. Sequences have different prior
lengths and distinct cache values; each page table is physically permuted.
The cache is packed C8 with per-128 scales and BF16 rotary. Zero query vectors
make attention weights uniform, allowing an independent analytic check of
every output, including the 128 sink tokens. Per-row sparse indices exclude
future positions and select at most 512 entries. This specifically checks
sequence boundaries, page-table isolation and length interpretation; it does
not replace nonzero-query numerical qualification or learned-index selection.

Control tests passed 17 negative cases, 12 single-sequence equivalence cases,
4100 mixed row positions and fixed/shared/ring page-table checks. Existing
page-binding tests passed and the full native engine rebuilt successfully.

Multi-sequence learned index selection, BF16 window attention, decoder batch
submission and frontend scheduling are still required. The production decoder
continues to supply one sequence; this is not an end-to-end batching claim.
