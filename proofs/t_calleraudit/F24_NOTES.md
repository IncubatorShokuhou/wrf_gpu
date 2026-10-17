# F24 — API-only land-surface schemes (slab 1, Noah-classic 2, Pleim-Xiu 7): caller notes

Scope: `sf_surface_physics` 1/2/7 are in scheme_catalog `_IMPLEMENTED` but the native/nested CLI refuses
them (io/namelist_binding.py ~:538-546); they run only through the Python API (operational_mode hooks
~:9271/:9308/:9340). No v-prep composite uses them. Source: lsm part audit (parts/lsm.md, helper HIGH,
both sides read); the precipitation items re-checked by the lane owner against pristine WRF.

## Class A — precipitation never reaches the land model
| scheme | WRF | port |
|---|---|---|
| Noah-classic (2) | module_sf_noahdrv.F:835 `PRCP=RAINBL(i,j)/DT`; :856-866 `FFROZP=SR(I,J)` because module_surface_driver.F:2708-2709 sets `frpcpn=.true.` whenever SR is present (always on the ARW call) | coupling/noahclassic_surface_hook.py ~:212 `prcp=zeros`, `ffrozp=zeros` |
| Pleim-Xiu (7) | module_sf_pxlsm.F:482 `PRECIP = MAX(1.0E-3*RAINBL(i,j)/DTBL, 0.0)` (m/s) | coupling/pleim_xiu_surface_hook.py ~:188-191 `precip_2d = 0` |
| slab (1) | SLAB has no precipitation input | n/a |
RAINBL = previous-step RAINCV + RAINNCV (+RAINSHV) accumulated over the PBL interval and reset after the
LSM (module_surface_driver.F:1568, :4539); with bldt=0 it is exactly one step's RAINCV+RAINNCV.
After F1/F2 the port already holds that amount as `NoahMPPrecipitation.prcpconv + prcpnonc` (mm/s) for
every admitted MP/cumulus scheme; the carry is just not seeded/passed for these three LSMs.

## Class B
- slab / PX: downward SW used where WRF passes net GSW (module_sf_slab.F:404, module_sf_pxlsm.F:596);
  lsm_slab.py:148, pleim_xiu_surface_hook.py:200-201.
- Noah-classic: Q2 passed as a mixing ratio where WRF passes specific humidity (module_sf_noahdrv.F:797
  `Q2K=QV3D/(1+QV3D)`); noahclassic_surface_hook.py:191. SFCPRS from the lowest level and an inferred CH
  instead of WRF's (noahdrv :795/:847).
- slab / Noah-classic / PX: air temperature from theta_m instead of dry t_phy (≈2–4 K) — same class
  as S-B (now owned by t-cfix-phyprep).
- slab FLHC/FLQC and PX RMOL recovered by inverting flux handles (slab_surface_hook.py:162-232,
  pleim_xiu_surface_hook.py:124-146) instead of WRF's surface-layer outputs.

## Fix options for the class-A items (decision requested from the manager)
1. **Fix (recommended for PX, cheap and exact):** seed the precipitation carry also for explicit
   Noah-classic/PX and pass `prcpconv+prcpnonc` (PX: ×1e-3 → m/s; Noah-classic: mm/s = RAINBL/DT).
2. **Noah-classic FFROZP needs WRF's SR**, the microphysics' own frozen fraction of RAINNCV. The public
   slots reproduce it for most schemes ((SNOWNCV+GRAUPELNCV[+HAILNCV])/(RAINNCV+1e-12)), but not
   Ferrier (passes RAINNCV + SR only; port keeps `state.ferrier_sr`) — and Ferrier+Noah is WRF's NAM
   pairing. A literal fix adds SR to the held precipitation carry (restart schema bump, E78) or reads
   each scheme's SR leaf (ferrier_sr, thompson_sr, goddard_sr, ntu_sr, cam_sr, generic diagnostics).
3. **Or fail closed** for API runs of sf_surface 2/7 (like the CLI already does) until 1+2 are done.
Recommendation: (1) for PX now; Noah-classic via (2) with an SR carry field in the same series; B items
to the S-B owner (theta) and this lane (SW/GSW, Q2) where cheap.
