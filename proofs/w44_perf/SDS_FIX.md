# ShapeDtypeStruct canonical-key repair

Candidate `2c941de611`, hardened by `69d404541f`, fixes a confirmed latent pointer-hashing bug. **It does
not explain H1 on the sampled PROD path.** The independent PP04 census found
zero SDS content-walker visits in eight valid PROD keys.

The descriptor branch precedes NumPy materialization and has its own typed
`shape-dtype-struct-v1` namespace. It encodes shape, dtype, weak_type and is_ref.
Placement, local layout and manual-axis metadata are refused until supported;
the dedicated exception propagates out of `_structure_fingerprint` so its coarse
fallback cannot erase a static descriptor determinant. `cheap_key` then returns
None and uses the normal lowering path. Concrete arrays retain their v0.4.3 byte
encoding. The global v5 key composition stays unchanged.

The successor also propagates this refusal through fn_identity_hash when a
CodeType constant holds an unsupported descriptor; its ordinary best-effort
exception path must not drop the code determinant. A sixth focused test covers
that path. The prior zero-SDS production traces cannot enter either refusal,
so their byte-equal key proof carries forward without repeating initialization.

Five focused CPU tests pass: separate equal descriptors; zero NumPy conversion;
fresh-process stability; shape/dtype/weak/ref closure-key separation against
actual lowered HLO; unsupported static pytree-aux refusal; prior concrete hash
goldens. The weak_type case is the independent reviewer's E45 counterexample.

Actual WN3 staged 0227 input files were read with the manager's standing w44
authorization and every file SHA was checked against the sealed v0.4.3 manifest.
Old and fixed walkers were applied to the same real initialized arguments,
through the authoritative runtime key probe. Six radiation-init keys and four
step keys (d01, d02, d03, fused/d02) are byte-identical; both walkers encountered
zero descriptors. Every comparison executed the walker and required valid keys.
The receipt is [WN3_SDS_KEYS.json](receipts/WN3_SDS_KEYS.json).

The 519-file traced-source closure excludes aot_cheap_key.py and its content
digest is identical in the baseline and candidate. The PROD PP04 zero-visit
trace therefore reuses the exact non-SDS execution/encoding proof; it needs no
duplicate model initialization or numerical validation. The repair changes no
model arithmetic, compiled layout, device work, buffers or IO. It adds one host
type check per canonical walk node and a small metadata tuple only for SDS.
No GPU speedup is claimed.

A once-only warm WN3 three-nest run is prepared for the manager's night
sequencer. It uses the unchanged v0.4.3 sealed native cache through its consumer
overlay and the actual candidate Python source; asserts GPU, fused/d02 load,
zero warm JAX cache misses, and the first twelve release-file hashes. This is a
regression/cache check, not original CPU-WRF fidelity. GPU execution **passed**
at 22:36:44Z on the actual final source: 12/12 complete NetCDF files byte-equal
to the release, 1660/1660 persistent cache hits, zero misses, and consumer
`compiled_new=0, aliases_new=0`. Fused d02+d03 loaded the sealed native program.
The GPU content-walker census observed 66,641 nodes and zero SDS descriptors.
Live device peak was 2,910,642,324 B. Its observed 47.57 s process duration is
diagnostic, not a speed claim. [Final receipt](receipts/SDS_FINAL.json).

Astra approved the final source subgate at 21:42:24Z: six independent
focused passes, three genuine assertion-mutant kills, key/source identity audit,
and no blocking findings. Review `105e24b326`, immutable verdict SHA-256
`79bc8e7c8af336e028fac8b96a8b5fc0d05780a04cbe3b9aba2d42fe7579471e`.
Native E41 is reused from the byte-identical sealed executable artifacts; no
cold compilation occurred. Private overlay, outputs and both now-unused SDS
snapshots were removed after the compact receipts and hashes were retained.
This is not an RW/fidelity or H1 speed verdict. The original timing session is
unchanged, and the main performance investigation remains open.
