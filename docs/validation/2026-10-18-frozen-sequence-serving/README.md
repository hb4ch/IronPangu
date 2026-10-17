# Frozen native sequence serving qualification

The same candidate backend completed discovery, fixed-workspace preparation and
frozen-arena execution in Docker `inferfabric-92b` on 7.156.99.58. All three
native processes exited 0; each matched all 159 reference output tokens across
21 requests through first EOS. All eight ranks report zero activation host
transfers. Fixed and frozen runs report planned workspace with no growth.

The final bundle is
`/data/p00603624/inferfabric/openpangu92/sequence-serving-frozen-plan`.
Its program and backend fingerprints were checked by the normal binding path;
none were rewritten to reuse evidence from another binary.

- Shared workspace per rank: 159,221,248 bytes.
- Frozen arena per rank: 38,116,810,768 bytes.
- Discovery arena extent: 38,259,441,168 bytes. Planning the maximum workspace
  first eliminates retention of obsolete workspace growth allocations.

`scripts/qualify-sequence-serving.py` reproduces the three stages, using existing
token fixtures and qualified reference output. Every stage must pass before
the next starts. Production handles now expose batch capacity after matching
arena/workspace validation and startup graph preparation. The previous
preparation-only batch restriction has been removed.

This qualifies the native mixed-request path with frozen resources. Rust HTTP,
prefix caching, longer multi-chunk prompts and 64K serving remain separate gates.
