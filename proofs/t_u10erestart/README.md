# Optional surface wind restart admission

`u10e/v10e` are equivalent-roughness winds produced by MYJSFC (surface option 2)
and QNSESFC (surface option 4). They legitimately remain `None` for other surface
schemes. SSiB's validation iterated over every history slot and rejected these
inactive fields. The fix resolves activity per field, retaining the existing
shape, REAL32 and finiteness checks on every required or populated field.
The same predicate covers QNSE PBL with a different surface scheme.

Active MYJSFC and QNSESFC restart admission also requires both winds. All three
public restart APIs reach this through the existing common surface admission.
Native storage uses its existing authenticated all-domain TEMF surface selector
when there is no QNSE-specific configuration. NetCDF validates raw stored history
before JAX conversion, so x64-OFF cannot hide a malformed float64 wind.
The wire representation and schema version are unchanged; no carry is invented.

The exact introducing patch is `fc37e012639219c22fb95060bdb07bdfd6c6956a`, also
present as rebased `b8d09d96893fbdb0269c3a441d5cb4b0a98324b2`. Both have stable
patch ID `a131fb040de2ffa06a268bb31d5d994a1bcc206e` and history source blob
`964054ee4c8217df5f9a0b0d37c90f4a4a260e83`. The actual SSiB pickle roundtrip
passes on immediate parent `7cd23caef1b199060b1263673febcc7766c1d186` (3.87 s)
and fails on `b8d09d968` (3.12 s) at `history.u10e`, with zero setup errors.
The validator blob `a489465f8e278f644494d61702a4d56676efbcca` and test SHA256
`eb9cad30db4062e0297913e403279571adf16eefd1d70e1308f9ef078ef7cddf`
are unchanged across that measured edge. Receipts: UR02/UR03 in the lane directory.

Release-default Swiss (`sf_sfclay_physics=5`, Noah-MP) bypasses SSiB validation:
`src/gpuwrf/io/restart_schema.py:519-520` returns when the selector is inactive
and both SSiB groups are absent (baseline main `9b394e500`: lines 494-495).
UR01's first full Swiss CLI attempt reached the imposed 6 GiB cgroup memory cap
during CPU compilation, before publication. After the nightly reported DONE and
the manager authorized 12 GiB singletons, the real default CLI gate passed.
The default CPU portable path has no held-history group; its winds are `None`.

CPU qualification records 500 unique passing checks, including all 15 original
SSiB failures on the composed main with t-redsuite's three test corrections.
Deleting the activity predicate causes 12 genuine assertion failures and zero
setup errors. Both complete default program hashes match: Swiss `f672a573…`
and PROD d01 `6fc715a2…`. The receipts and exact-source reuse proof are retained
in `receipts/`. These are CPU admission/regression and source-identity checks,
with no GPU execution or WRF-fidelity claim.

The worker gate is green; the same critic's delta review and manager merge are
next. The last MP38 smoke node passes in 25.45 s with 6.73 GiB peak RSS under the
12 GiB cgroup cap, closing all 104 shared physics nodes. The prior sealed
`receipts/FINAL_CPU_RECEIPT.json` remains unchanged; `receipts/FINAL_DELTA.json`
closes its deferred gates without repeating the 499 checks or default identities.

Both real Swiss CLI cases use 42×42×44, dt 18 s: uninterrupted control to step
200, verified checkpoint and three exports at step 100 followed by self-SIGKILL,
then fresh-process resume to step 200. Default surface/PBL 5 preserves all 155
terminal carry leaves, 754 history variables across two files and global
attributes byte-for-byte. MYJ surface/PBL 2 preserves all 163 terminal leaves,
758 variables and attributes. Native storage and all three standalone formats
(pickle, NetCDF, M7 checkpoint) restore each complete step-100 carry exactly.
Both actual MYJ winds are finite float32 42×42 arrays with 1,764 nonzero entries;
their ranges and hashes are sealed in `receipts/cli_myj/verification.json`.

These are CPU restart/determinism gates with `GPUWRF_FAST_DEFAULTS=0`, cache
disabled, MemoryMax=12 GiB and SwapMax=0. The borrowed real CLI probe redirects
only State's device guard to CPU. The new wrapper asserts the loader's effective
physics before compilation. A discarded MYJ attempt revealed that separate
`--namelist` affects preflight but does not reach the native loader
(`cli.py:1197-1215`, `nested_pipeline.py:934-950`); the valid MYJ case instead
places its regular namelist inside a private input directory with the exact
frozen real Swiss input files. No CLI product code was changed. No WRF fidelity
or GPU execution claim is made. Bulk outputs and the detached snapshot were
removed after SHA256 inventories; the 14 MB private input case is retained.
All product bytes match reviewed source `c2d131cd8` (gpuwrf tree
`276c2da5153997d6916057fbbed9e83c32a33a2a`).

Performance scope: only restart-boundary host code changes. No timestep solver,
compiled model input, device kernel or carry layout changes. The inactive path
adds a Python selector check; active winds add two host surface-array validity
checks. NetCDF defers its existing JAX conversion without introducing another
copy. Measured on NumPy arrays outside the timestep: at 117×267, inactive
admission is 0.23 µs/64 tracked temporary bytes; active MYJ admission is
5.67 µs/32,305 bytes (1,000 samples, shared host). At 42×42, active admission
is 2.85 µs/2,830 bytes. No GPU speed or VRAM claim is made.
