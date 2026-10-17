# Packed cache control staging — 2026-10-17

The executor stages all cache append positions in one preallocated control bank
per rank. The compiler-defined instruction order assigns 16-byte slots: INT64
physical position at offset 0 and its checked INT32 form at offset 8. Descriptor
views retain fixed addresses across graph replay and page-table rebinding.
Host metadata contains positions only; cache contents remain device resident.

For the current 46-cache 92B plan this replaces 92 small per-token uploads with
one **736-byte upload**. Viewer recipes expose the bank size and each instruction's
byte offset; a compiler test verifies all 46 non-overlapping slots and page sizes.
The per-request block tables are still bound separately after reset. Future batch
staging can extend this control bank without per-layer host/device transfers.

Full TP8/EP8 C8 qualification matched all **99 expected tokens** with independently
shuffled page tables. Fresh discovery and fixed-workspace preparation passed before
binding. Short-prompt TPOT is **17.826–17.929 ms** per request, compared with
20.156–20.495 ms before staging consolidation. This recovers the preceding serial
baseline; it is not a production throughput benchmark.

The rank-0 control-input audit fell from **22,615 to 2,959 H2D calls** for the same
nine-request workload, including startup and rebinding. Transfer bytes rise slightly
because the bank includes padding and both integer views. Activation host transfers
remain zero. Frozen arena capacity is **12,408,072,256 bytes/rank** and workspace
remains **130,916,352 bytes/rank**, with zero workspace growth. Summarized native
audits retain original hashes; step records cover the final request, while graph
replay counts and transfer audits cover the engine lifetime.

Reproduce with the existing fresh workspace/arena preparation sequence, followed
by `qualify-program-engine.py --permuted-pages` and `qualify-paged-boundaries.py`.
Old frozen plans are invalid for this backend binary and must be regenerated.
Native compilation and all device execution ran in Docker; Rust compiler builds
ran locally. Compiler tests passed.

The same-engine identity/shuffled comparison also passed **224 tokens across 28
requests**, with prompt lengths 1–504 and eight output tokens per request. At 504
prompt tokens, TTFT is 9.234–9.244 seconds and TPOT 18.674–18.685 ms. The preceding
implementation measured 10.391–10.518 seconds and 20.933–21.201 ms respectively.
These are synthetic repeated-token layout tests under serial prefill, not external
model-quality evidence or 64K qualification. The boundary harness also checks
frozen arena/workspace, zero activation host transfers and no token allocations
or internal completion fences in retained step records.

This improves metadata staging only. Shared native state, multiple recurrent
slots, true batched decode, packed/chunked prefill, learned DSA for 64K and hybrid
prefix reuse remain release work.
