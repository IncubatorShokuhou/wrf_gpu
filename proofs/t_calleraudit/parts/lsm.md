# t-calleraudit — land surface (sf_surface_physics 0/1/2/3/4/7/8) caller wiring vs WRF 4.7.1

Static, read-only audit (grep/sed only; no python, pytest or GPU). Port tree = worktree `t-calleraudit` (lane/t-calleraudit,
b51dd0805). WRF = `<USER_HOME>/src/wrf_pristine/WRF`. Paths are abbreviated: `om` = src/gpuwrf/runtime/operational_mode.py,
`nmc` = physics/noahmp_coupler.py, `nmh` = coupling/noahmp_surface_hook.py, `sd` = phys/module_surface_driver.F,
`nmd` = phys/module_sf_noahmpdrv.F. Release defaults come from src/gpuwrf/_fast_defaults.py (Q2 cap flag :139, Noah-MP julian :109).
Composite map, taken from v-prep namelists: Noah-MP (4) in c00, c02–c38, c41–c59; RUC (3) in c33 (mp8 cu0 MYNN); SSiB (8) in c40
(mp8 cu0 sf1 bl1); lsm 0 in c01/c39 (no sfclay). Slab (1), Noah-classic (2) and PX (7) have no composite and are API-only:
the CLI refuses them (namelist_binding.py:248, NATIVE_LAND_OPTIONS=(0,3,4,8)).

## 0. Ranked findings

| # | cls | scheme(s) | one line | conf |
|---|---|---|---|---|
| A1 | A | 4 (+urban SLUCM/BEP inside it) | Noah-MP gets **zero precipitation** (rain, snow, graupel, hail, convective) for every mp_physics ∉ {5,7,8,15,38}. This hits 24 composites (c02–c25). | HIGH |
| A2 | A | 4, 3, 8, urban | **Convective precipitation is dropped** (PRCPCONV/RAINCV=0) for cu ∈ {2 BMJ, 3 GF, 5 G3, 6 Tiedtke, 16 nTiedtke, 93 GD}. | HIGH |
| A3 | A | 2, 7 | Noah-classic PRCP and PX PRECIP are hard-wired to 0. WRF passes RAINBL. | HIGH |
| B1 | B | 1, 2, 7 | Air temperature is built from **theta_m** (moist θ) instead of WRF's dry t_phy/th_phy: about 2–4 K too warm. | HIGH |
| B2 | B | 1, 7 | **Downward SW** is used where WRF uses net GSW (slab RNET; PX GSW slot). | HIGH |
| B3 | B | 2 | Noah-classic Q2K is passed as a mixing ratio (WRF uses specific humidity). SFCPRS and CH are proxies, not WRF's values. | HIGH |
| B4 | B | 1, 7 | Slab FLHC/FLQC and PX RMOL are recovered by inverting flux handles, with clips. PSFC is the lowest-level p (WRF uses p_hyd_w(1)). | HIGH |
| B5 | B | 4 (opt_sfc=2 + sf=4) | Noah-MP CH input is overwritten by the QNSE CH. WRF passes CHXY (Noah-MP's own previous CH). | MED |
| B6 | B | 3, 8 (slope_rad=1) | RUC/SSiB GSW/SWDOWN do not get WRF's TOPO_RAD_ADJ slope correction. Noah-MP does, via swnorm. | MED |
| C1 | C | 0, 1, 3, 8 | WRF's final land cap `Q2=MIN(Q2,1.05*QV1)` (sd:4546-4558, applied after any LSM) exists only in the Noah-MP coupler. | HIGH |
| C2 | C | 4 | Over urban/LCZ land with sf_urban_physics=0, WRF T2/Q2 = T2MB/Q2MB; the port writes the FVEG blend. | MED |
| C3 | C | 3 | Water QSFC: WRF LSMRUC overwrites it with qvg/(1+qvg) and SFCDIAGS uses that. The port keeps the MYNN QSFC (history Q2 over water). | MED |

## 1. Noah-MP (sf_surface_physics=4, release default)

(a) **Default branch.** `om:9047-9196`, the `use_noahmp` branch:
- The sfclay runs inside the adapter first (`nmc:298-303`, MYNN-SL `surface_layer_with_diagnostics`), unless sf1/epsilon/QNSE/TEMF diagnostics are handed in.
- It then calls `noahmp_surface_step` (`om:9112-9145` → `nmh:214-344` → `nmc:223-533`).
- Radiation default: `radiation_diagnostics` is present, so `(swnorm | swint swdown, glw, coszen)` is used (`om:9064-9068`).
- Precipitation comes from the `noahmp_precipitation` carry. It is written after the post-RK MP (`om:10384-10405`) and read at the next step.

**WRF call site.** `sd:3146-3285` (CALL noahmplsm); the forcing setup is `nmd:732-795`; the post-LSM T2/Q2 logic is `sd:3438-3473` and `sd:4546-4558`.

(b) Port vs WRF, item by item:

| item | WRF | port | verdict |
|---|---|---|---|
| order radiation → sfclay → LSM | `first_rk_step_part1` radiation, then `sd`: sfclay select :2007ff, then LSM :3146 | `om:8957-8973` radiation refresh, then sfclay inside `nmc:298-303`, then Noah-MP | MATCH |
| T3D | `T_ML=T3D(1)=t_phy` (dry θ·Π from p+pb), `nmd:752` | `nmh:141-147`: θ_m→θ_dry, then T from nonhydrostatic p | MATCH |
| QV3D | `Q_ML=QV/(1+QV)`, `nmd:753` | `nmc:115-116` | MATCH |
| U/V | `U_PHY(1)/V_PHY(1)` mass point | `nmh:165-166` `_u_mass/_v_mass` | MATCH |
| P8W | `P8W=grid%p_hyd_w` (first_rk_step_part1:637). `P_ML=(P8W(2)+P8W(1))/2` = p_hyd(1); `PSFC=P8W(1)`, `nmd:757-759` | `nmh:149-155` p_hyd(1) (sfcprs) and p_hyd_w(1) (psfc) | MATCH |
| DZ8W/ZLVL | `0.5*DZ8W(1)`, g=9.81, `nmd:749` | `physics_couplers.py:1624-1633` g=9.81; `nmc:176-178` | MATCH |
| SWDOWN | held, slope-adjusted (`sd:~1900` TOPO_RAD_ADJ) or swint | `om:9064-9068` `max(swnorm,0)` / swint swdown | MATCH |
| GLW / COSZEN | held; COSZEN only at radiation calls, at `xtime+radt/2` (radiation_driver:1207-1209) | held `rad.glw` / `rad.coszen` | MATCH |
| SWDDIR/SWDDIF | passed, but used only by noahmp_urban | not passed (urban uses its own inputs) | MATCH (n/a without urban) |
| RAINBL/MP_* (Thompson, 5/7/15/38) | `RAINBL+=RAINCV+RAINNCV(+RAINSHV)` (`sd:1568-1570`, reset `sd:4531-4544`). `MP_RAINC=RAINCV`, `MP_RAINNC=RAINNCV`, `MP_SNOW=SNOWNCV`, `MP_GRAUP`, `MP_HAIL` (`sd:3283-3284`) → `/DT` (`nmd:778-788`). Values are the previous step's MP plus the held cumulus amount. | `noahmp_precipitation.py:16-42`: total/dt, (snow+ice)/dt, graupel, hail, spec-zone mask (B40). Convective = held KF PRATEC (`kf_rk.py:58-62`). Time level n-1. | MATCH |
| same, other MP or cu | as above | A1 / A2 | **MISMATCH** |
| SR / PRCPOTHR | `PRCPOTHR=max(0, PRCP-CONV-NONC-SHCV)`, which is 0 without fog/irrigation | not formed (irrigation and fog are refused) | MATCH |
| RAINSHV | MP_SHCV | shcu refused (namelist_binding.py:480-482) | n/a |
| JULIAN/YR/ITIMESTEP | `JULIAN_IN=grid%julian`; `soiltstep=0` → `calculate_soil` every call (`nmd:648-675`) | `_noahmp_clock` (`om:6847`, `NOAHMP_JULIAN_ADVANCE=1`); soiltstep>0 refused (scheme_catalog.py:722) | MATCH |
| TMN/VEGFRA/SHDMAX/IVGTYP/ISLTYP | statics | traced statics (B41, B65) | MATCH (known) |
| CMXY/CHXY in | `CM=CMXY`, `CH=CHXY` (`nmd:817-818`); pure outputs for opt_sfc=1 (SFCDIF1) | `nmc:322-324` seed `_get(diag,'ch',land.ch)`. MYNN diagnostics have no `ch` → land value kept. | MATCH (opt_sfc=1); B5 |
| sea ice | `XICE>=thr`: CYCLE, then seaice_noah | refused: land_history.py:67, lower_boundary.py:86 | MATCH by refusal |
| water points | CYCLE (`nmd:732`), so sfclay values stay | `nmc:428-436` where(is_land) | MATCH |
| outputs TSK/HFX/QFX/LH/QSFC/ZNT | grid fields read by the next sfclay/PBL | `nmc:428-490`; QSFC/HFX/QFX/MOL carried (B39) | MATCH |
| ALBEDO/EMISS → radiation | grid%albedo/emiss | `physics_couplers.py:1215-1227` reads land_state.albedo/emiss | MATCH |
| T2/Q2 water | bulk form (`sd:3446-3458`) | `nmc:545-558` (B49) | MATCH |
| T2/Q2 land | FVEG blend; urban/LCZ and ISICE&XICE<thr → T2MB/Q2MB (`sd:3459-3473`) | `energy.py:1082,1103,1146`: blend if veg & fveg>0, no urban branch | C2 |
| Q2 cap | `sd:4556` | `nmc:536-542` (flag default ON) | MATCH |

(c) Findings

**A1 — Noah-MP receives no precipitation unless mp_physics ∈ {5,7,8,15,38}.** HIGH: both sides were read.
- Port, the carry is seeded only for that list:
  `om:7793-7795`: `if (use_noahmp or ruc or ssib) and run_physics and int(namelist.mp_physics) in (DEFAULT_MP_PHYSICS, 5, 7, 15, 38): ... seed_precipitation`
- With the carry absent, three things follow:
  - the next-step update is skipped (`om:10384`, `if carry.noahmp_precipitation is not None`);
  - no `precipitation=` kwarg is passed (`om:9123-9124`);
  - `assemble_noahmp_forcing` falls back to zeros (`nmc:151-155`): `return _surface(v) if v is not None else jnp.zeros(shape...)`.
- Also, mp=28 and every `MP_SCAN_ADAPTERS` scheme return no precipitation (`om:8579-8582`).
- WRF: `RAINBL(i,j)=RAINBL(i,j)+RAINCV(i,j)+RAINNCV(i,j)` (`sd:1568`) and `MP_RAINC=RAINCV, MP_RAINNC=RAINNCV, ... MP_HAIL=HAILNCV` (`sd:3283-3284`). Then `PRCP=PRECIP_IN/DT` and `PRCPNONC=MP_RAINNC/DT` (`nmd:764,779`).
- Effect: no canopy interception, infiltration, runoff or snowpack from precipitation. Soil and canopy dry out, which biases LH/HFX/T2. The urban SLUCM/BEP `rainbl` is also 0 (urban_slucm_hook.py:198, urban_multilayer_hook.py:116).
- Composites affected: c02–c25 (mp 1,2,3,4,6,9,10,13,14,16,17,18,19,21,22,24,26,28,40,50–53,97).
- The same hole was already found and refused for RUC (`om:7386-7391`, "any other MP would silently feed zeros (rv-ruc #4)"). The release default LSM did not get that refusal.
- Fix:
  - make every admitted MP adapter return its per-step RAINNCV/SNOWNCV (snow+ice)/GRAUPELNCV/HAILNCV amounts. Most already accumulate RAINNC.
  - seed the carry whenever any LSM with a precipitation input is active.
  - until then, refuse Noah-MP + those MPs.
- Oracle: a pristine WRF 1-step run on a raining column, with the MP=6/10 RAINNCV dump vs the Noah-MP forcing `PRCPNONC*DT`. Then a 1-h Swiss composite: SMOIS(1)/CANLIQ/SNEQV vs CPU-WRF (c06/c08 D6 on SMOIS).

**A2 — Convective precipitation is dropped for the stateless and BMJ cumulus schemes.** HIGH.
- Port: `om:10399` `convective_rate = jnp.zeros_like(next_state.t_skin) if rates is None else rates[6]`.
  - `cumulus_tendencies` is only seeded for cu 1,4,14,94,95,96,99 (`om:7691-7724`); cu 7/10/11 use cuplus PRATEC (`om:10400-10401`).
  - cu 3/5/6/16/93 (CU_STATELESS, `om:9762-9765`) and BMJ cu 2 (`om:9814-9822`) only add RAINCV into `rainc_acc` (scan_adapters.py:1024-1026, 1155-1157, 1313-1315, 1412-1414).
- WRF: RAINCV is held from the last cumulus call (module_cu_tiedtke.F:403, module_cu_gf_wrfdrv.F:802, module_cu_bmj.F:257). It feeds RAINBL (`sd:1568`) and MP_RAINC (`sd:3283`); for RUC, `rainbl-rainncv` is the convective liquid/frozen part (module_sf_ruclsm.F:624-632); for SSiB, `raincv(i,j)` is passed directly (`sd:4360`).
- Affected combos: Noah-MP / RUC / SSiB / urban with cu ∈ {2,3,5,6,16,93}, e.g. RAP-like RUC+GF+Thompson+MYNN (RUC admits any cu), or Thompson+Tiedtke+Noah-MP. No composite isolates this: c03/04/05/06/07/13/17 are already A1.
- Fix: export the held RAINCV/DT (or PRATEC on call steps, held between calls, per WRF's RAINCV=rn/stepcu semantics) from these adapters into `precipitation_from_step`.
- Oracle: a WRF Tiedtke column RAINCV vs the port PRCPCONV·DT over one stepcu cycle.

**B5 — opt_sfc=2 with sf=4.**
- `QnseSurface.ch` (qnse_adapter.py:49) reaches `nmc:322-324` as `ch_seed`, so Noah-MP's CH is replaced by the QNSE CH.
- WRF passes `CHXY` (`nmd:818`), which SFCDIF2 uses at iteration 1 (`AKHS` in module_sf_noahmplsm.F SFCDIF2: `WSTAR2=...AKHS*DTHV`).
- Only opt_sfc=2 (admitted, noahmp_options.py:22) + sf=4 is affected. c26 uses the default opt_sfc=1, so it is unaffected.
- Fix: do not seed CH/CM from any sfclay diagnostics; always pass land_state.ch/cm.

**C2 — Urban T2/Q2.** WRF `sd:3459-3467` sets `T2=T2MBXY` for ISURBAN/LCZ (and ISICE&XICE<thr). The port blend (`energy.py:1103`) has no urban branch, so urban cells without urban physics get FVEG·T2MV+(1-FVEG)·T2MB. The same applies to Q2.
- History only, but it biases urban T2 station scores.
- Also: on non-ISWATER water cells (MODIS lake 21, XLAND=2), WRF writes the never-updated `FVEGXY*T2MVXY...` blend while the port writes the sfclay T2. Port is arguably better; disclose.

## 2. RUC (sf_surface_physics=3; composite c33)

(a) **Default branch.** `om:9351-9396`: native radiation gives `GSW=swdown-swup` (`om:9359-9361`), then `ruc_surface_step` (ruc_surface_hook.py:146-280). Admission: mp=8, sfclay=5, MYJ refused, sea ice and lakes fail closed (`om:7381-7400`; namelist_binding.py:730-735).

**WRF call site.** `sd:3541-3700` (CALL LSMRUC `sd:3603`; CHS/CQS for SFCDIAGS `sd:3683-3688`; SFCDIAGS_RUCLSM `sd:3691`).

(b) Port vs WRF:

| item | WRF | port | verdict |
|---|---|---|---|
| sfclay → LSM, FLHC/FLQC | MYNN-SL first; LSMRUC `qkms=flqc/rho/mavail`, `tkms=flhc/rho/(cp(1+.84qv))` (ruclsm.F:684-686) | `surface_layer_with_exchange` exact FLHC/FLQC (hook:166-169) | MATCH |
| CHS in | used only if myj (ruclsm.F:681-682) | zeros; MYJ refused | MATCH |
| t_phy/qv/qc/P/rho/dz8w | `p_phy` = grid%p_hyd(1) (`patm=p8w(1)*1e-5` slot), rho, dz8w (`sd:3612-3613`) | hook:176-185 `t_air`, `p_hyd(1)`, phy_prep rho, dz8w | MATCH |
| GSW/GLW | GSW net (RRTMG SWDNB-SWUPB), slope-adjusted if slope_rad=1 | `rdiag.swdown - rdiag.swup` (unadjusted) | MATCH (B6 if slope_rad=1) |
| RAINBL/RAINNCV/SNOWNCV/GRAUPELNCV/SR | RAINBL=RAINCV+RAINNCV; Thompson `SR=(snow+graupel+ice)/(RAINNCV+1e-12)`; frpcpn=.true. | hook:114-136: `rainbl=conv+nonc`, `rainncv=nonc`, `snowncv=snow+ice`, `frz=(snow+ice+grpl)/(nonc+1e-12)` | MATCH (mp8); A2 for cu |
| ktau | itimestep (the first-step block only) | `where(first_timestep,1,2)` | MATCH |
| outputs → next sfclay/PBL | TSK=soilt, ZNT, MAVAIL, QSFC, HFX/QFX/LH | hook:218-247 blend on land | MATCH |
| ALBEDO/EMISS → radiation | grid fields | `om:8116-8117` `ruc_radiation_surface(carry.ruc_land)` | MATCH |
| water points | LSMRUC sets `QSFC=qvg/(1+qvg)` with `qvg=qsn(soilt)/patmb` (ruclsm.F:836-843) and HFX/QFX are held; SFCDIAGS uses it | port ruclsm.py:1809-1818 computes it, but the hook keeps the MYNN `diag.qsfc` on water (hook:243-244) | C3 |
| T2/Q2/TH2 | SFCDIAGS_RUCLSM, then the `sd:4556` land cap | ruc_diagnostics.sfcdiags_ruclsm, no 1.05·QV cap | C1 |

(c) Findings. A2 applies to RUC with cu ∈ {2,3,5,6,16,93}; c33 is cu0, so it is unaffected.
- **B6:** with slope_rad=1, WRF adjusts GSW before LSMRUC (`sd:~1895-1913`). The port uses `swdown-swup` (`om:9361`). Use the topographic pair (`swnorm - swup_topographic`, as ocean_surface_hook.py:16 does) or refuse slope_rad with RUC.
- **C1/C3:** history-only. MYNN-SL recomputes water QSFC on the next call, so there is no prognostic effect.
- Oracle for C3: the c33 CPU-WRF water Q2/QSFC field vs GPU.

## 3. SSiB (sf_surface_physics=8; composite c40)

(a) **Default branch.**
- The sfclayrev runs at `om:9206-9216`: `sfclay_revised_mm5_adapter(..., surface_inputs=ssib_surface_inputs)`.
- Then `ssib_surface_step` (`om:9336-9350`, ssib_surface_hook.py:52-110).
- YSU then gets the SSiB operands: `ssib_ysu_step` (`om:9514-9516`, hook:155-181).
- Admission: mp8, sf1, bl1, ra_sw1, ra_lw4, 3 soil layers, fractional_seaice=1 (namelist_binding.py:724-729; `sd:1485`). Sea ice refused (ssib_land_init.py:48).

**WRF call site.** `sd:4259-4440`, plus the land-point HOLD restore after sfclayrev at `sd:6209-6227` (SFCLAYREV_SEAICE_WRAPPER).

(b) Port vs WRF:

| item | WRF | port | verdict |
|---|---|---|---|
| precip | `rainncv(i,j), raincv(i,j)` (amounts) | `ppl=prcpnonc·dt`, `ppc=prcpconv·dt` (hook:79) | MATCH (mp8); A2 for cu |
| atmosphere | `u_phytmp(1)`, `v_phytmp(1)`, `qv_curr(1)`, `t_phy(1)`, `p_phy(1)` = p_hyd, `psfc` = p_hyd_w(1), `dz8w(1)` | hook:76-82 via `ssib_physics_inputs` (hook:127-147, dry θ, p_hyd, g 9.81) | MATCH |
| radiation | swdown, glw, coszen; cloudfrac = max_k min(CLDFRA,1) (`sd:4328-4331`) | Dudhia SWDOWN (albedo-0 solve, hook:203), glw, coszen held; cloud = max_k of held CLDFRA | MATCH |
| xlat/day | `xlat_urb2d=grid%XLAT` (first_rk_step_part1:771); `day=int(julian_in+.01)+1` | `static.xlat` (XLAT degrees); `ssib_day` (hook:33-48) | MATCH |
| outputs | TSK, HFX, QFX, LH, UST, ZNT=ssib_z00, BR, GZ1OZ0, T2=TSK, U10/V10 from UV10 | hook:91-109 + `ssib_ysu_step` br=land.rib | MATCH |
| HOLD restore (itimestep>1) | land BR/T2/Q2/HFX/QFX/LH/GZ1OZ0/WSPD/ZNT/UST restored after sfclayrev; FM/FH from sfclayrev | WSPD restore (hook:96-98); SSiB values overwrite the rest; YSU psim/psih = sfclay fm/fh | MATCH (reviewed in t-lowersweep c40) |
| Q2 | land cap at `sd:4556` | `q2=new.q2m`, no cap (hook:103-104) | C1 |

(c) Findings. A2 for cu ∈ {2,3,5,6,16,93} (c40 is cu0). C1 history Q2. B6: SSiB uses unadjusted `rdiag.swdown` when slope_rad=1; low priority.

## 4. Noah classic (sf=2), slab (sf=1), Pleim-Xiu (sf=7): API-only, no composite (brief)

**Branches.**
- Noah-classic: `om:9248-9266` → noahclassic_surface_hook.py.
- Slab: `om:9267-9303` → slab_surface_hook.py:235ff.
- PX: `om:9304-9335` → pleim_xiu_surface_hook.py.

All three take radiation from the legacy held `_refresh_noahmp_rad` tuple: soldn = max(swnorm,0), i.e. DOWNWARD SW (`om:8275`).

**WRF call sites.** SLAB `sd:2646-2685`; Noah `lsm(...)` `sd:2922` with noahdrv; PXLSM `sd:3701-3804`.

| item | WRF | port | verdict |
|---|---|---|---|
| air T | `t_phy` (dry) for SLAB/lsm/PXLSM; PX also `th_phy` | `_temperature_from_theta(state.theta, state.p)` with state.theta = θ_m (physics_couplers.py:502-506): slab:257, noahclassic:164/189, PX:174/277; PX `theta1=state.theta[0]` (PX:175) | **B1** |
| SW | SLAB `RNET=GSW+XINET`, GSW net (module_sf_slab.F:404; radiation_driver:3785). PX `SOLDN=GSW/(1-ALBEDO)` plus net GSW (pxlsm.F:596,639). Noah `SOLNET=SOLDN*(1-ALBEDO)` (noahdrv:834) | slab gsw = downward (lsm_slab.py:148). PX passes the downward value as both SOLDN and GSW (PX:200-201). Noah: soldn·(1−static alb) (noahclassic:196-210) | **B2** (slab, PX); Noah ≈ (static vs snow albedo) |
| precip | Noah `PRCP=RAINBL/DT` (noahdrv:835); PX `PRECIP=1e-3*RAINBL/DTBL` (pxlsm.F:482) | `prcp=zeros` (noahclassic:212); `precip_2d=zero2d` (PX:188-191) | **A3** |
| humidity | Noah `Q2K=QV/(1+QV)` (noahdrv:797) | `q2=max(state.qv[0],0)`, a mixing ratio, labelled specific (noahclassic:191; lsm_noah_classic.py:133) | **B3** |
| pressure | Noah `SFCPRS=(P8W(2)+P8W(1))/2`=p_hyd(1) (noahdrv:795); SLAB/PX `PSFC=p8w(1)`=p_hyd_w(1) (`sd:1988`) | lowest-level nonhydrostatic `state.p[0]` (noahclassic:190, slab:270, PX:177) | B3/B4 |
| exchange | SLAB FLHC/FLQC from the sfclay; Noah `CHK=CHS` (noahdrv:847); PX `rmol`/`ust` from the sfclay | inverted from flux handles with clip/fallbacks: slab:162-232, noahclassic:161-178, PX:124-146 | **B4** |
| CQS2 | SLAB: `CQS2=CQS2*MAVAIL` before SFCDIAGS (`sd:2651-2655`) | UNVERIFIED | C |

These are moot while the CLI refuses 1/2/7, but the API path is advertised.
- Fix order: θ_m→θ_dry decoupling (copy `nmh:141-147`); net GSW for slab and PX; precipitation carry for Noah-classic and PX; `QV/(1+QV)`; p_hyd/p_hyd_w; take FLHC/FLQC/CHS/RMOL directly from the sfclay diagnostics (as RUC does) instead of inverting handles.
- Oracle: pristine single-column SLAB/SFLX/PXLSM drivers fed by the port's caller arrays vs WRF's.

## 5. No LSM (sf_surface_physics=0)

WRF `CASE DEFAULT` (`sd:4456-4462`) calls nothing:
- land TSK stays at its initial value;
- water TSK follows SST (sst_update);
- HFX/QFX/QSFC come from the selected sfclay with MAVAIL;
- RAINBL is still accumulated and reset;
- the Q2 cap (`sd:4546-4558`) still applies over XLAND<1.5.

Port: the else-branch runs only the sfclay adapter (`om:9201-9247`), with no LSM write-back, so t_skin is unchanged → MATCH.
- With sfclay ∈ {1,91,3,7}, the land Q2 cap is absent (C1). MYNN-SL applies its own internal 1.05 cap (surface_layer.py:1023).
- c01/c39 run sfclay=0, so no surface fluxes at all — MATCH (WRF returns at `sd:1490`).

## 6. Proof plan (cheapest first)
1. A1/A2: a CPU-only unit test asserting `NoahMPForcing.prcpnonc·dt == RAINNCV` for each admitted MP, and `prcpconv·dt == RAINCV` for each admitted cu. It must be deletion-sensitive (E39) and run on the real `_physics_step_forcing` seam, not the adapter.
2. Then c06 (WSM6+Tiedtke) and c08 (Morrison) 24 h CPU-WRF vs GPU SMOIS/CANWAT/SNOW/LH maps. They should show a dry-land drift today.
3. C1: c33/c40 history Q2 vs CPU-WRF on land cells where Q2 > 1.05·QVAPOR(1).
