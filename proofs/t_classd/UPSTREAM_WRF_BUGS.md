# Upstream WRF 4.7.1 observations (t-classd)

## MSKF_REFL_10CM reads unallocated cumulus arrays when aercu_opt=0

`module_mp_morr_two_moment_aero.F:970-973` copies `NR_CU/NS_CU/QR_CU/QS_CU(i,k,j)` into
`NRMSCU1D/...` for every column, and the `diagflag .and. do_radar_ref==1` branch
(`:1193-1203`) adds `QRMSCU1D*CU_UAF(i,j)` (likewise NR/QS/NS) before the second
`refl10cm_hm` call that produces `MSKF_REFL_10CM`. Those arrays and `CU_UAF` belong to the
Registry package `aercuopt` (`aercu_used==1`, Registry.EM_COMMON:3255), which is inactive
for `aercu_opt=0` — the only value the port admits (`aercu_opt>0` needs MSKF `cu_physics=11`
plus CESM aerosol input). Inactive package state is allocated with extent 1, so these reads
are out of bounds in WRF itself (undefined values).

Port policy (standing compiler-defined-zero policy for undefined reads): the cumulus
contribution is zero, so with `do_radar_ref=1` the port writes `MSKF_REFL_10CM` equal to
`REFL_10CM`. No CPU-WRF reference exercises this branch (all composites run
`do_radar_ref=0`, where WRF never assigns `MSKF_REFL_10CM`).

## NaN radar for mass without number (refl10cm_hm, both Morrison modules)

`refl10cm_hm` (`module_mp_morr_two_moment.F:4544-4580`) forms the slope from `nr/rr`
etc. whenever the mass exceeds 1e-9 kg/kg. A column with `qs > 1e-9` and `ns = 0`
gives `lams = 0`, `ilams = Inf`, `N0_s = 0` and `ze_snow = 0*Inf = NaN` in pristine WRF
(observed in an early synthetic oracle column). Morrison's own number clipping keeps
numbers positive with mass, so the operational path does not produce this state; the
port reproduces the same arithmetic.
