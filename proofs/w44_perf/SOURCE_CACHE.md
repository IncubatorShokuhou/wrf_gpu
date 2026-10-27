# Content-addressed source import cache

Candidate `2b8c56012f` removes repeated AST scanning from warm process startup.
The H1 caller measurement found 519 source files and 3.55 million AST nodes;
the first source fingerprint took 3.078 s on CPU, before the first SW init
kernel's cache lookup. This is separate from the 1,094 MLIR lowerings.

Every lookup reads and SHA-256 hashes the current source **bytes**. The cache
key combines that hash with an algorithm version; its value stores raw import
specifications. Relative imports are resolved for the current module after
lookup, so identical file bytes in different packages remain correct. The
closure walk and final full source-byte digest still run every process.
Neither mtimes nor sizes substitute for content. A change to extraction
semantics requires a cache algorithm version bump.

Entries have a payload checksum and atomic publication. Missing, truncated,
corrupt, incompatible or unwritable entries fall back to parsing. The cache
lives under the existing configured compile-cache directory; disabling that
cache disables this metadata cache too. No new environment key or AOT key
schema is introduced. Cache misses use a statement traversal that also visits
nested blocks/functions/classes and skips expression subtrees, which cannot
contain Python import statements.

| Fresh CPU process, same immutable source [M] | Fingerprint mean s | Whole probe process mean s | AST parses |
|---|---:|---:|---:|
| Actual legacy walker, 3 processes | 2.7463 | 3.0507 | 519 each |
| Candidate, metadata cache disabled, 3 processes | 1.7915 | 2.0931 | 519 each |
| First metadata population, 1 process | 1.9079 | 2.2132 | 519 |
| Warm metadata component, 3 processes | 0.0564 | 0.3545 | 0 each |
| Actual configured production cache path, warm, 3 processes | 0.0592 | 0.3419 | 0 each |

All ten component samples and all four configured-path samples return the
same fingerprint as the original v0.4.3 algorithm. The measured component
saving is 2.690 s per fresh process; the production resolver adds about 3 ms
to the warm component. The metadata cache has 519 entries, 314,269 payload
bytes. Peak RSS in the component probe falls from 360,948 to 167,037 KiB.
The actual file allocation is disclosed in the production-path receipt.
These are CPU caller/probe measurements, not end-to-end GPU forecast speed.
No new device operations, transfers or device allocations are introduced.

Qualification: 21 focused tests pass in 35.67 s, including the existing six
SDS tests. Another 84 consumer checks pass in 28.03 s, with five existing
skips. The checks compare the full real package against the legacy AST walk,
full-tree cold/cached/legacy fingerprints and AOT keys, a one-byte edit with
identical size/mtime (exactly one reparse, changed fingerprint), newly reachable
imports, relative package contexts, four corruption cases, interrupted atomic
publication, unwritable storage, algorithm versions and concurrent writers.

The exact-source proof confirms all 519 trace-reachable files are byte-equal
to the sealed v0.4.3 files; the key module itself is outside this closure.
Every other key algorithm is AST-identical. The final source fingerprint
`e14f436be7775f292af0bcc240c4093cf3b0aa253c0b0414208254da060428e9`
matches every recorded real WN3 step key's source component. Existing real
PROD/WN3 key and native validation therefore carries by determinant identity;
no new real initialization/key census or WRF-fidelity run is claimed.

[Timing receipt](receipts/SOURCE_CACHE_TIMING.json),
[configured-path receipt](receipts/SOURCE_CACHE_PRODUCTION.json),
[exact key/source proof](receipts/SOURCE_CACHE_KEY_PROOF.json).
Astra approved **MERGE** at 2026-10-24 00:01:34Z, review `bb2bf67667`.
Independent full-tree/edit/context/corruption and two-process atomic-publication
checks pass; two assertion mutants are killed with no setup failures. Verdict
SHA-256: `fd4916ba49547a07a9c308f58f74f4d85ab6d4f8d53511d2f820406da93d368d`.
Product commit `2b8c56012f` is integrated in main (verified at `d1db02bd62`,
with the exact reviewed module SHA). The source-cache subgate is green; the
main performance investigation and three-version README
comparison remain open. Review snapshot and disposable metadata caches are
removed after the critic released their last dependency.
