# F2-FIX + F24 — convection-only land forcing; Noah-classic / Pleim-Xiu precipitation

Lane t-calleraudit · product commit 76860e2d1 (on top of F2 412e0eeff) · CPU only.

## 1. F2 FIX (rv-t-calleraudit verdict 18:38Z)
Defect: with mp_physics=0 and cu 2/3/5/6/16/93 the land carry was never seeded (seed required mp≠0)
and, if seeded, `_apply_post_rk_microphysics(return_precipitation=True)` returned a bare State for
mp=0 → PRCPCONV stayed 0. WRF: RAINBL += RAINCV + RAINNCV for every scheme set
(module_surface_driver.F:1568), RAINNCV/SR keep their Registry 0 without microphysics.
Fix: `_land_precipitation_seed(namelist)` seeds for every land consumer (Noah-MP, RUC, SSiB, PX →
"plain"; Noah-classic → "sr") independently of mp; mp=0 returns zero RAINNCV/SNOWNCV/GRAUPELNCV/SR.
Test: the actual `_physics_boundary_step` for cu 2/3/5/6/16/93 × mp {6, 0} ends with
PRCPCONV = f32(.36)/f32(dt) and the production seed predicate returns "plain" for the Noah-MP copy.

## 2. F24 (manager decision 18:1xZ: fix PX + Noah-classic, SR in the held carry, restart bump)
Defect (CALLER_AUDIT F24, API-only LSMs): Noah-classic PRCP/FFROZP and PX PRECIP were hard-wired 0.
WRF: module_sf_noahdrv.F:835 PRCP=RAINBL/DT; :859-866 FFROZP=SR (FRPCPN=.true. because SR is present,
module_surface_driver.F:2708); module_sf_pxlsm.F:482 PRECIP=MAX(1.0E-3*RAINBL/DTBL,0).
Fix:
- `NoahMPPrecipitation.sr` optional (default None): held only for Noah-classic; the Noah-MP/RUC/SSiB
  release carry is leaf-identical (None is an empty pytree node); `noahmp_surface_hook` passes only
  the five PRCP* slots to the column view.
- SR = the microphysics' own WRF SR: generic adapters from kernel `diagnostics['sr']` (WSM3/5/6/7,
  WDM5/6/7, Lin, Goddard97; Morrison/Morrison-aero now export theirs), explicit producers (Ferrier SR,
  Goddard-4ice, Thompson-hail, NSSL, Milbrandt, NTU, CAM-MG; P3 requests its kernel `sr` output only
  when Noah-classic is selected), Thompson mp 8/28 literal mp_gt_driver :1308 form, Kessler/SBU/Jensen
  0 (WRF never sets SR for them); any other producer without SR fails closed.
- `noahclassic_precipitation`: PRCP = max(PRCPCONV+PRCPNONC, 0), FFROZP = SR; `pleim_xiu_precipitation`:
  max(1e-3·(PRCPCONV+PRCPNONC), 0); both refuse a missing carry (E78), Noah-classic also a missing SR.
- Restart: restart.py FORMAT_VERSION 6, pre-v6 checkpoints with Noah-classic land refused (E78);
  wrfrst/nested restores of a 5-field tuple yield sr=None → refused by Noah-classic at first trace;
  SSiB validator and WRF-restart hydration accept/hydrate the optional channel (hydration: WRF `SR`).

## 3. Evidence [M CPU]
- Pristine REAL4 statement oracle (proofs/t_calleraudit/f24_statement_oracle.json, built by
  build_f24_statement_oracle.py from the untouched WRF statements, 288 rows): port PRCP ≤ 1 ULP
  (224/288 exact), PX PRECIP ≤ 2 ULP (180/288 exact) — the carry holds RAINCV/DT and RAINNCV/DT
  separately (Noah-MP's interface), so the sum of rounded quotients differs from WRF's
  (RAINCV+RAINNCV)/DT by ≤ 1 ULP (disclosed); FFROZP = SR exact.
- tests/v040/test_t_calleraudit_f24_lsm_precip.py (21) + land_precip (F1/F2/F2-fix, 50) = 71 PASS (corrected count).
- Consumers: 106 test files touching the changed seams (restart pkl/wrfrst/hydration, SSiB, RUC, lake, Noah-MP
  hook, every modified producer, physics dispatch), all via RAM slots: 2,313 PASS, 15 SKIP; 3 FAIL all
  PRE-EXISTING on main c874dcc13 (test_aot_baked_constants external-reader disposition cammg/ntu tables;
  test_v040_urban_lcz multilayer [2][3] — identical on main). Two test pins updated to the producer's actual
  contract (nssl_cumulus mock now returns `rainncv`/`sr` like nssl2mom/driver.py:99; test_cammg key set
  + `sr`). tests/v025/restart/test_cpu_wrfrst_ozone_hydration.py SKIPS (P0227 savepoints absent) → the
  WRF-restart hydration edit (zip of the five WRF slots + optional SR) is not exercised by a runnable test.
- Mutants (F24_MUTANTS.json): 6/6 killed by genuine assertions, 0 setup errors — seed skips mp=0
  convection-only (the F2 FIX case), Noah-classic drops SR, PX precipitation zero, restart v6 refusal
  removed, Thompson SR without ice, generic scheme SR ignored.
- Default identities EXACT: Swiss f672a573, PROD d01 6fc715a2 (F24_SWISS/PROD_IDENTITY.json).
- Operational smoke tests/test_v013_operational_smoke.py: 93/93 PASS, one RAM slot per test (91 within 6 GiB;
  the mp38 Thompson-hail and mp56 NTU microphysics tests need >6 GiB and passed at 12 GiB: test footprint,
  pre-existing). Receipt F24_SMOKE_PERTEST.txt.

## 4. Performance / memory
Release path unchanged. Noah-classic runs: one extra REAL (ny,nx) carry leaf (4 B/column) and one
division; P3 + Noah-classic: one extra kernel output. Non-default MPs carry an unused `sr` dict entry
(DCE'd).

## 5. GPU reruns
None beyond the F1/F2 lists (no composite uses Noah-classic/PX or mp=0 with an LSM and cumulus).
