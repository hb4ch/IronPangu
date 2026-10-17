# Compressed prefix index: host metadata qualification

The runtime now has a generic compressed token radix index in
`crates/inferfabric-runtime/src/prefix_index.rs`. This is a prerequisite for the
hybrid device prefix cache, not an enabled serving cache or an NPU output result.

The index compares exact token spans and the full `PrefixIdentity`, including
tenant, executable, checkpoint, backend, topology, tokenizer/template and adapter.
Checkpoint selection additionally compares completed arithmetic history. A split
creates metadata only: it cannot invent a complete recurrent/cache checkpoint.
Lookup reports token-match length separately from the deepest eligible checkpoint.
The caller supplies a maximum restore boundary; using prompt length minus one
preserves final-logit recomputation, falling back to an earlier complete checkpoint.

Metadata admission checks entry count, conservative token-storage budget, exact
history length and page alignment before mutation. Removal prunes empty branches
and recompresses edges. IDs are unique across indexes and never reused within an
index. Failed insertion and stale/foreign removal preserve existing entries.

Validation ran locally in WSL:

```
cargo test -p inferfabric-runtime --test prefix_index
```

All four tests pass: split/branch/full-hit behavior, identity/arithmetic/generation
isolation, 65,536-token capacity failure behavior, and repeated branch removal
checked against a linear exact-token oracle.

The index contains no device handles or tensor contents. Its caller must own and
pin complete sealed state, select unpinned LRU victims, retire device resources,
reserve private restore storage and execute explicit native copies before admitting
a restored request. Those integrations and cold/hit/branch NPU comparisons remain
required. No hit rate, latency improvement or production readiness is claimed.
