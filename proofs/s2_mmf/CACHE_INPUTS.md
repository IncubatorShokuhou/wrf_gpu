# MMF cache inputs

`opt_run=5` adds the conditional `noahmp_mmf` namelist marker; `opt_inf` is
in both the namelist aux tuple and the MMFBundle static aux. The generic
`aot_cheap_key` dataclass/aux traversal therefore includes both selectors.
Invariant ACRECH/ACQSPRING zero metadata is also in the bundle aux.

FDEPTH, AREA, HGT, EQZWT, RIVERBED, RIVERCOND, PEXP, SMOISEQ, RECHCLIM,
eligibility, FSATMX, WTDDT seconds and STEPWTD are runtime array leaves.
The new `_fsatmx` file reader is not cached; its value enters the resident
bundle as an explicit REAL4 argument. It creates no baked table provider.
Existing Noah-MP parameter-table readers remain under the shared cache policy.
The literal 19-entry lateral-conductivity factors belong to the source
fingerprint, matching the original WRF constants.

`tests/test_noahmp_mmf_cache.py` checks actual MPTABLE FSATMX replacement,
unchanged HLO/key for runtime values, changed finite runoff and byte-identical
reuse versus fresh compilation. Its static-selector test checks distinct
merged-helper keys and HLO for opt_inf=1/2 and disabled MMF. These checks are
cache regressions; original-WRF fidelity is established by separate fixtures.
