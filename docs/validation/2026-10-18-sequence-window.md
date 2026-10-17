# Multi-sequence BF16 window attention prerequisite

Production metadata generation and window attention now use shared helpers in
`resident_window_attention.h`. Their query ends, KV lengths and page tables are
explicit tensor inputs, permitting independent sequences in one invocation.
The existing decoder supplies the same single-sequence descriptors as before.

On 7.156.99.58 inside Docker `inferfabric-92b`,
`native-build-head-batch/sequence_window_probe` returned:

```json
{"passed":true,"batches":6,"checked_values":25288704}
```

The probe calls both production helpers. Each batch packs three sequences:
one decode at position 65535, an extend starting at 32767, and one cold token.
Total query lengths are 3, 17 and 4096, tested with windows 128 and 512.
Each sequence owns a reversed physical page mapping and an independent ring;
the page table repeats its own ring over the logical 64K context. The fixture
materializes the entire live window-plus-chunk union before attention and
poisons other storage. This checks causal exclusion of already-written future
tokens, window expiration, sequence isolation and ring wrap.

Zero query vectors give uniform weights, enabling an analytic oracle over the
correct window and 128 sink tokens for every output. This checks addressing
and lengths; it is not nonzero-query numerical or full-model 64K qualification.
Fixtures and readback are test-only. No long prefill graph is captured.

The full native engine rebuilt successfully. End-to-end multi-request execution
still requires decoder control wiring, per-request sampling, native submission
and the frontend scheduler. No serving performance claim follows from this probe.
