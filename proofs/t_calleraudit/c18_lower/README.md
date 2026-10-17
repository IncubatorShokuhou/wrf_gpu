# FINAL c18–c24 (+c65, c88) d01 lowering TypeError — F8 passive number scalars on the release MYNN entry

Lane t-calleraudit · rebased on main db78f9bd8c (batch58) · CPU only (no GPU used).
Review range: `db78f9bd8c..HEAD` of lane/t-calleraudit. Product commits (after rebase): 9153f96074 (fix + test),
6aefb36fd6 (lower-error traceback), 89e5dfc2c7 (test process isolation); src diff = phys_mynn_plume.py + domain_tree.py
only. The three older branch commits on top of main are notes/proofs only (no src).
Evidence in §4 was measured on the pre-rebase heads 710ea6e12c / 716d292d3a (base 3871ee9f37, tag
t-calleraudit/c18fix-pre-rebase); batch58 touches only Tiedtke/scan_adapters src. Composed head re-checked [M CPU]:
F20 MYNN caller 29, traceback 2, passive-native 9 PASS (6G slots, `COMPOSED_db78_TESTS.txt`); Swiss 7379c7d7 / PROD
6487f5b0 EXACT (`IDENTITY_*_composed_db78.json`).

## 1. Defect [M CPU]
v-prep FINAL-10/11 on 04cb09f816: c18 (mp26), c19 (mp28), c20 (mp40), c22–c24 (mp51/52/53), all bl5/sf5/Noah-MP/RRTMG/cu0,
model_rc 1 at d01 lowering: `TypeError: float() argument must be a string or a real number, not 'dict'` (no traceback printed).
CPU repro with v-prep's exact c18 input + environment (GPUWRF_FAST_DEFAULTS=1, CENSUS=1, RRTMG_MP_RE=1, JAX_PLATFORMS=cpu):
full traceback `c18_cpu_traceback_04cb.txt` →
`_physics_step_forcing` → `mynn_adapter_with_source_leaves(passive_scalars=…)` → `step_mynn_pbl_column_with_pblh` →
(GPUWRF_MYNN_FP32_COLUMNS) `phys_mynn_columns._advance_native` → `_native_edmf_arrays(**_ni_plume_kwargs(state))` →
`phys_mynn_plume.dmp_mf_columns_native`: `jnp.asarray(v, float32) for every kwarg` → `passive` is a dict → TypeError.

## 2. Attribution
dac4a66631 (F8 "MYNN mixes/applies QNC, QNWFA, QNIFA", lane t-cfix-pblmix, merged batch51 ff17ba6a2d) added
`_ni_plume_kwargs` → `{'passive': state.passive}` and the `passive=` branch to the RETAINED `mynn_edmf.dmp_mf_columns`
only. Its "native" tests went through `mynn_edmf.dmp_mf_columns` (which peels `passive` before dispatch) and the
whole-entry test ran `_advance_native(edmf=False)` (plume never called) → the production entry was never exercised.
de374a3c00 (c18 GPU rc 0) predates it. The trigger is MYNN × an MP with WRF number scalars
(`_pbl_mynn_number_scalars`: Nc for mp 9/11/14/16/26/28/38/40/51/52/53/56, + nwfa/nifa for 28/38) — not the species
path itself; mp 17/18/19/21/22/24/30/32/50 are not affected by this cause. Active FINAL composites hit: c18 c19 c20 c22
c23 c24 c65 c88 (= F8's own "GPU rerun after merge" list).

## 3. Fix (`src/gpuwrf/kernels/phys_mynn_plume.py`, `dmp_mf_columns_native`)
`passive` is peeled off; each species is evaluated on the existing native QNI lane of an otherwise identical plume
(`_dmp_mf_columns_impl(..., qni=species)["s_awqni"]` → `s_awq<name>`), exactly F8's retained design (WRF DMP_mf runs the
QNI passive entrainment recurrence + limiter for UPQNC/UPQNWFA/UPQNIFA, module_bl_mynnedmf.F :6099-6103, :6172-6176,
:6411-6420). Passive lanes do not feed back. Empty/None `passive` → identical trace (release mp8).
Separate commit 710ea6e12c: both nested-AOT lowering sites (`_lower_advance_variant`, `_lower_fused`) print the full
traceback to stderr via `_lower_error_text` (one-line status unchanged; fail-open unchanged).

## 4. Evidence [M CPU]
- Lower-only CLI harness (`repro_lower.py` + `run_lower.sh`: real `gpuwrf run` on v-prep's exact case input + env,
  stops right after `_lower_advance_variant` digests the d01 step program; private caches deleted per case):
  BEFORE main 3871ee9f37: 8/8 TypeError (c18 c19 c20 c22 c23 c24 c65 c88, 32–53 s each) `LOWER_BEFORE_main3871ee9f37.txt`;
  AFTER fix 710ea6e12c: 8/8 lowered on CPU (70–87 s each, hlo digests recorded) `LOWER_AFTER_fix710ea6e12c.txt`.
- `tests/test_t_calleraudit_mynn_passive_native.py` (9): native entry vs F8's pristine DMP_mf fixture (16 real CPU-WRF
  c19 columns, QNI 5 % gate) for s_awqNc/nwfa/nifa; each species == a separate native QNI-lane call BITWISE; all base
  outputs bitwise unchanged; real Swiss root d01 step LOWERS under the release MYNN flags for mp 26/28/40/51/52/53/38/56
  (one fresh process per MP, peak 3.5–3.9 GB) and the native entry received + returned the Registry set.
  Whole file 9 PASS in one 6G RAM slot (7:56). `tests/test_t_calleraudit_lower_error_traceback.py` (2) PASS.
- Mutants `MUTANTS.json` (`mutants.py`, private snapshot, files restored, genuine signatures, 0 setup errors): see §6.
- Consumers (snapshot 710ea6e12c; src identical to 716d292d3a), one 8G slot per file: F8 suite 25, F20 MYNN caller 29,
  stochastic SPP 7, tests/v025/b_phys MYNN (closure/condensation/dheat/elb_mf/exner/held diag/native plume/phy_exner/
  plume cloud base/real coverage/plume sums/saw floor/QNI/real constants/scale aware/sfc wspd) 691, domain_tree 28,
  cheap-key memo 6, nested runtime reuse 12, parallel compile 17, aot_cheap_key 99 → 914 PASS, 0 F/E; tree clean after.
- Release identity (proofs/s1_port/release_identity.py, location-stripped jaxpr): Swiss 7379c7d7…aec55e,
  PROD d01 6487f5b0…2c34e73 EXACT (`IDENTITY_SWISS.json`, `IDENTITY_PROD.json`).

## 5. Performance / memory
Release default (mp8, no number scalars): unchanged program (identity EXACT). Nc-carrying MPs (opt-in): per MYNN call
1 (Nc) or 3 (mp 28/38) extra native plume-sums evaluations (extra `fused_plume_sums` launch + assemble; the identical
setup is CSE-able by XLA [I]); GPU time not measured [U]. Same cost model as F8's retained path. A multi-lane passive
kernel would remove it (follow-up if these composites matter for speed). No carry/State change.

## 6. Deletion sensitivity
M1 revert (passive passed to jnp.asarray) → TypeError in both the oracle test and the mp28 root-step lowering;
M2 species lane keeps the QNI values → oracle 5 %/bitwise assert; M3 passive dropped → KeyError 's_awqNc' / assert;
M4 helper prints no traceback → assert; M5 fused site bypasses helper → AST assert.

## 7. GPU reruns after merge
c18 c19 c20 c22 c23 c24 c65 c88 (v-prep HOLD set). Suggested preflight: `run_lower.sh <tree> <tag> <cases…>` (CPU, ~75 s/case).

## 8. Disclosures
- Only the passive-plume oracle is a WRF comparison; the whole-step check is lowering only (no numerics, no GPU).
- MF is active in 3/16 of the F8 January columns (F8's own disclosure).
- Lowering on CPU interprets Pallas; Triton lowering of the extra plume-sums calls is the same kernel as the QNI lane
  already in production (mp8 carries Ni) — not separately compiled here.
