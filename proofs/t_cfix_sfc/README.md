# t-cfix-sfc — surface layer -> LSM/PBL handoff (CALLER_AUDIT F3 / F12 / F13 / F19 / F21)

Lane t-cfix-sfc, 2026-10-15. Pristine reference: WRF 4.7.1 `<USER_HOME>/src/wrf_pristine/WRF`
(unmodified lane-dir copies, E104). No GPU work in this lane; [M CPU] unless noted.

## What changed (WRF caller semantics)
- **sf 3/7/91** (`coupling/sfclay_selected.py`): ONE selected surface call per step
  (SF_GFS / PXSFCLAY / SFCLAY, module_surface_driver.F:2344/2165/2064) on the qualified SF1
  phy_prep operands (dry th/t_phy, p_hyd, PSFC = p8w(kts), dz8w g=9.81) with WRF grid INOUT
  memory (UST ZNT MOL QSFC HFX QFX PBLH; PX itimestep). Its outputs are Noah-MP's
  `surface_diagnostics` (no MYNN-SL re-solve, F3), no-LSM history, and BL_GFS's
  PSIM/PSIH/BR/WSPD/GZ1OZ0 (F13). Kernels gained opt-in WRF surface_driver constants
  (cp = 7 r_d/2, ep_1, ep_2 — E83) and REAL sfclayinit tables; legacy defaults are unchanged.
- **bl=3 requires sf=3** (WRF isfc=2; module_physics_init.F:3743 is fatal otherwise; c25 = sf91+bl3
  died in CPU-WRF with exactly that message).
- **BL_GFS** (critic FIX 22:54Z): the selected-surface GFS PBL call is WRF's: REAL grid operands
  copied into a narrow kind_phys DOUBLE island (`scan_adapters.gfs_bl_real_double`), REAL stores,
  and the INOUT grid writes UST=SQRT(STRESS), WSPD=SQRT(U(kts)**2+V(kts)**2)+1.E-9
  (module_bl_gfs.F:464-466); the next SF_GFS call reads that UST.
- **sf=0** (F21): no surface layer and no land model (module_surface_driver.F:1490 returns first).
- **sf 2 / bl 2** (`coupling/myj_surface.py`, `physics/myj_adapters.py`): ONE MYJSFC call per
  step with carried THZ0/QZ0/UZ0/VZ0/AKHS/AKMS (NTSD==1 cold start), TKE operand = qke/2;
  MYJPBL reads this solve, ELFLX = post-LSM LH, CHKLOWQ = 1 after an LSM, SNOW = Noah SWE, and
  returns THZ0/QZ0/QSFC (F12). The memory rides the Janjic-family `qnse` carry leaf
  (QNSESFC is the MYJSFC derivative with the identical INOUT set; E78 restart schema shared).

## Oracles (each: zero / inf / FMA builds; zero == inf byte-identical)
| file | routine | columns | calls | gate |
|---|---|---|---|---|
| `sfclay91_oracle.f90`, `sfclay7_oracle.f90`, `sfclay3_oracle.f90` | SFCLAY, PXSFCLAY, SF_GFS | 238 (CPU-WRF c04 lowest level + water/calm variants) | 2 chained | port inside WRF's own +-1 ULP input envelope (256 draws + FMA build) +2 output ULPs |
| `myjsfc_oracle.f90` | MYJSFC (+MYJSFCINIT) | 252 (CPU-WRF c03 44-level profiles + water/calm) | 3 chained (itimestep 1 cold) | REAL mode, 128 draws, +8 ULP (123/79380 cells need 3-8 ULP, disclosed) |
| `blgfs_oracle.f90` | BL_GFS -> MONINP | 84 c04 profiles fed by the pristine SF_GFS (+/-QI), plus the ACTUAL operational caller operands | 1 | 64 draws, +8 ULP, subnormals flushed (E117) |

Scripts: `build_oracle.py`, `make_oracle.py` / `make_oracle_myj.py`, `ulp_envelope.py` /
`envelope_myj.py`, `compare_port.py` / `compare_myj.py`, `make_fixture.py` / `make_fixture_myj.py`,
`mutants.py`. Tracked fixtures: `fixtures/sfclay_wrf_oracle.*`, `fixtures/myjsfc_wrf_oracle.*`.

## Known, disclosed residuals
- SFCLAY regime-1 keeps the ENTRY ZOL (WRF INOUT); the port does not hold ZOL (no consumer).
- PSIHTB entries N=1 and N=922 differ by 1 ULP from the gfortran/glibc REAL table (logf rounding).
- MYJ without an LSM: WRF never sets the i1 field CHKLOWQ; zero policy (UPSTREAM_WRF_BUGS.md).
- Non-ISWATER water points (lakes, no lake model) under Noah-MP keep the scheme's bulk 2-m form
  where WRF reads a skipped Noah-MP point (audit C2).
