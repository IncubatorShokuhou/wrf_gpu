# Surface analysis nudging

**Status in v0.4.0: Validated 24 h.** Passed a 24-hour comparison with the original CPU WRF in the v0.4.0 release check, in one tested single-domain configuration. Other combinations and nested domains are not validated.

`grid_sfdda=1` selects WRF PSU surface analysis nudging. `grid_sfdda=2`
selects its FASDAS atmospheric branch. Both require `grid_fdda=1`,
`bl_pbl_physics=5`, `sf_sfclay_physics=5`, and Noah-MP
(`sf_surface_physics=4`). WRF rejects surface plus spectral nudging
(`grid_fdda=2`). Observation nudging may coexist using its independently
validated settings and input stream.

Both options are also checked on the CPU against WRF's own routines, including
short stepping and exact restart checks. These short checks are not a full
hindcast validation.

```fortran
&fdda
 grid_fdda = 1,
 gfdda_interval_m = 60,
 gfdda_end_h = 24,
 guv = 0.0003, gt = 0.0003, gq = 0.0003,
 grid_sfdda = 1,
 sgfdda_inname = 'wrfsfdda_d<domain>',
 sgfdda_interval_m = 60,
 sgfdda_end_h = 24,
 guv_sfc = 0.0003, gt_sfc = 0.0003, gq_sfc = 0.0003,
 rinblw = 100.,
/
```

The default surface filename is WRF's `sgfdda_d<domain>`; the example selects
`wrfsfdda_d01`. `sgfdda_interval` is the fallback when
`sgfdda_interval_m=0`. Intervals are minutes, end times hours from the run
start, nudging coefficients s⁻¹, and `rinblw` km. `fgdt` controls the shared
FDDA recomputation cadence, including held tendencies and `fgdtzero`.
The surface and three dimensional streams may have different intervals.

Use NetCDF (`io_form_sgfdda=2`) with WRF `Times` and OLD/NEW bracketing
records. The first record must be at the model start; later records follow
the configured surface interval. Each record has `U10_NDG_OLD/NEW` on X
faces, `V10_NDG_OLD/NEW` on Y faces, and mass-shaped `TH2_NDG_OLD/NEW`
(absolute dry potential temperature, K), `Q2_NDG_OLD/NEW` (mixing ratio,
kg kg⁻¹), and `ODIS_NDG_OLD/NEW` (observation distance, km). Required fields
must be finite and complete. The loader rejects missing records, incorrect
shapes, masked data, and nonzero begin/unsupported interval components that
WRF's FDDA driver rejects.

WRF scales surface winds under free convection, smooths the observation
density weight once, and restricts surface moisture targets to its saturation
mixing ratio. Surface wind and theta innovations reference the lowest model
level throughout the PBL; moisture innovations are level-specific. The
three dimensional nudging terms are added above the PBL with WRF's transition
weights. The resident analysis pair and alarm cursor survive output segments
and checkpoint/restart.

FASDAS with classic Noah (`sf_surface_physics=2`) is refused because its
indirect soil/skin nudging kernel is not ported. Pristine WRF Noah-MP does
not consume those indirect FASDAS fluxes; mode 2 with Noah-MP follows that
behavior. `pxlsm_soil_nudge` remains refused because the PX land-model path
is not validated here. `HFX_FDDA` is exported as the held three-dimensional WRF heat-nudging diagnostic.
History-only `SDA_HFX/SDA_QFX`, `QNORM` and `HFX_BOTH/QFX_BOTH` are not exported.

Evidence and reproducible tools: `proofs/s2_sfdda/`.
