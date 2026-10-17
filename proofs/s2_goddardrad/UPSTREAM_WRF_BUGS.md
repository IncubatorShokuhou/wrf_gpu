# WRF 4.7.1 Goddard radiation reference dispositions

Source: `phys/module_ra_goddard.F`, SHA256
`0917531726cbed5a128e05c45e70ee292c209dbcd1ae2bad17db6e46b8d3a5ad`.
All numerical oracles use the original physics bodies in a lane-private build.
Private helper visibility shims change access only. The public-driver oracle
uses the original module byte for byte, including its REAL4/DOUBLE boundary.

## Undefined cloudy-SW fast-overcast optical depth

`swrad` declares `taux(CHUNK,np)` at line 3261 and reads it in the cloud-albedo
sum at line 5315. It passes the array to `sw_uvpar`, whose dummy declaration at
5493 has no assignment. Neither routine defines that array before the read.

Manager standing policy, explicitly confirmed for this lane on 2026-10-13:
build the pristine oracle with `-finit-real=zero`, and mirror zero at exactly
this read in the port. Do not replace it by a physically inferred optical
depth, change an existing formula, or modify pristine physics bodies.

G07 diagnostic: normal and zero-initialized builds complete clear and cloudy
columns. A signaling-NaN initialization with diagnostic invalid-operation traps
passes the clear column and traps at the original line 5315 in the cloudy one.
G10 is a separately labelled **sensitivity diagnostic, never a pristine oracle**:
a lane-private copy explicitly defines only the otherwise undefined `taux=1`.
Clear outputs are byte-identical; cloudy maximum flux-fraction difference is
0.0031552761793136597 [M]. The reference therefore depends materially on the
undefined local. Option 5 is experimental and discloses the chosen zero policy.

Reproduction: `probe_sw_definedness.py WRF BUILD`; the `one` arm is the explicit
site-only diagnostic, other arms have visibility-only source changes. Primary
receipts: `<USER_HOME>/wrf_gpu2_lanes/s2-goddardrad/G07_build/manifest.json`,
`manifest_one.json`, `snan_cloud1.txt` and `sensitivity.json`.
The operational SW heating correction at 2137–2141 is retained literally;
`sw_heating_negative_count` counts its events before zeroing on device.

## Exact-equator climatology out-of-bounds read

At latitude exactly 0, `sounding_interp` computes `inorth=9,isouth=11`
(2486–2487), then reads `mcdat(:,11,:)` and `(:,12,:)` from a ten-profile array
(2491). The bounds-checked original fails; this has no defined WRF value.
Operational loading explicitly refuses exact-equator cells. No interpolation
clamp or invented tropical profile repairs this source error. The internal
helper returns nonfinite data for such a bypassed call, rather than silently
using JAX's default clamped out-of-bounds gather.

Receipt: `<USER_HOME>/wrf_gpu2_lanes/s2-goddardrad/G01_build/equator.txt`.
