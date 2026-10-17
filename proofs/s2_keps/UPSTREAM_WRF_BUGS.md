# WRF 4.7.1 EEPS/KEPS undefined reads

Pristine source: `<USER_HOME>/src/wrf_pristine/WRF`, tag `v4.7.1`.
Bodies are copied byte-for-byte into lane-owned builds; the pristine tree is
never modified. Manager decision (2026-10-13): the qualification oracle uses
`-finit-real=zero -finit-integer=0 -finit-logical=false`, with floating-point
traps off. The JAX port explicitly mirrors zero at the sites below. Sensitivity
to alternative compiler initialization is reported; affected schemes are
experimental if the choice materially changes results.

| File / routine / line | Defect | Zero semantics |
| --- | --- | --- |
| `phys/module_bl_keps.F`, `GET_PBLH`, 822–823 | Reads implicitly declared REAL `ttke`; line 821 assigned `tke`, not `ttke`. | `ttke=0`; the maximum-TKE search never updates its initial maximum/index. |
| `phys/module_bl_keps.F`, `surface_bl_pra_ri`, 1065–1068 | Accumulates/normalizes `radsum` without initialization. | Begin the radiation sum at zero. |
| `phys/module_bl_keps.F`, `c_nsquare`, 660–668; consumed in `terms_tpe`, 1139–1142 | Assigns bottom/interior `dtmdz` but not `dtmdz(kte)`. | Top gradient is zero. |
| `phys/module_bl_eepsilon.F`, `eeps2d`, 426 / 748 | Declares `radsum(:)` but first accumulates into it without initialization. | Begin each column's top-down cooling sum at zero. |

Evidence already reproduced on unchanged sources:
`proofs/s2_keps/source_probe_receipt.json` contains compiler/source hashes,
SIGFPE backtraces at KEPS 822, KEPS 1068, and active-cloud EEPS 748, and an
explicit sentinel retained at the unwritten KEPS top gradient. Those diagnostic
poisoning runs identify undefined reads; they do not claim that production WRF
necessarily produces nonfinite forecasts.

Source SHA256:

- KEPS: `a79058405affcd293740b15137002e788d168a5eba51586d14cee41342ffc204`
- EEPS: `b1ea6db55866af15cb3c6962b1a15fa512eda803078f7304e6a133462d7e12de`
- WRF model constants: `5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062`

Full-driver sensitivity measured (KE13, unchanged source, REAL32, 60 carried
records per arm, both arms finite, traps OFF): zero vs sNaN initialization
materially changes both schemes. Largest examples from stratocumulus columns:

| Scheme / output | Maximum absolute difference | Zero-arm relative scale | Step |
| --- | ---: | ---: | ---: |
| EEPS TKE | 0.1087514626 m2 s-2 | 16.28% | 5 |
| EEPS heat diffusivity | 10.81948694 m2 s-1 | 99.80% | 5 |
| EEPS theta tendency | 4.07628249e-5 K s-1 | 8.99% | 4 |
| KEPS TPE | 0.000841984773 K2 | 807.01% | 5 |
| KEPS Prandtl number | 3.308358312 | 330.84% | 3 |
| KEPS heat diffusivity | 4.586938381 m2 s-1 | 81.75% | 5 |

The alternate compiler flag initializes **all** undefined local REAL storage;
this is a whole-driver sensitivity test, not an attribution of every output
difference uniquely to one defect. sNaN values can be hidden by later WRF
MIN/MAX operations, so finite output does not establish insensitivity.
Per-record evidence: `KE13/sensitivity.json`, `full_zero.json`, `full_snan.json`
in this lane directory. Source/build scripts and zero fixtures are tracked under
`proofs/s2_keps/`. Both port options must be labelled **experimental with
compiler-zero semantics**. No UCAR report has been sent.
