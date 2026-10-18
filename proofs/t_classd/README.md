# t-classd — class-D wrfout fields (v0.4.0 "not computed" list)

CPU qualification against pristine WRF 4.7.1 and original CPU-WRF composite outputs
(v-prep case dirs, read-only). No GPU work; GPU confirmation via v-prep later.

| Field(s) | Configuration | Implementation | Evidence |
| --- | --- | --- | --- |
| `REFL_10CM` | NSSL one-moment `mp_physics=19/21` | `nssl2mom.diagnostics.radardd02_one_moment` (radardd02 `ipconc=0` branches), held with the NSSL `makediag` lifetime like mp=18 | Pristine S5->S6 column fixtures (s2-mpmore, 17 cases x 2 variants): fp32 BYTE 680/680 levels each, fp64 <= 2.7e-16 rel. Original CPU-WRF c14/c15 history recompute, 25 frames: 7.6e-6 dBZ with WRF's dry microphysics density (`REFL_CPU_HISTORY_*_dry.json`) |
| `QGR_URB4D` | BEP+BEM `sf_urban_physics=3` | Held `QGR_URB3D` in `UrbanMultilayerCarry.green_roof`: Noah-MP SMOIS top layer at init, 0 on BEP/BEM columns after each call; restart E78 | c57 every cell of all 25 frames BYTE (`URBAN_CPU_ALL_FRAMES.json`) |
| `TGR_URB4D`, `DRAINGR_URB3D`, `EP_PV_URB3D` | BEP+BEM | WRF's +0 (init + per-call reset; PV power `pv_frac_roof`-weighted); green roof/PV refused at load in v0.4.0 (v0.4.1 ports both branches: [proofs/w41_writergaps](../w41_writergaps/README.md)) | same proof, +0 bits |
| `QNBCA` | `mp_physics=28/38` | WRF's +0 (`mp_gt_driver` zeroes NBCA unless `wif_input_opt=2`); nonzero QNBCA input refused (mp28 now, mp38 before) | c19/c65 all frames +0 (`CONSTANT_CPU_ALL_FRAMES.json`) |
| `TAOD5502D` | `mp_physics=28/38` | WRF's +0 (assigned only under `aer_opt=3`, refused) | c19/c65 all frames +0 |
| `MSKF_REFL_10CM` | `mp_physics=40`, `do_radar_ref=0` | WRF's +0 (assigned only under `do_radar_ref=1`) | c20 all frames +0 |
| `REFL_10CM` (+ mp40 `MSKF_REFL_10CM`) | Morrison `mp_physics=10/40`, `do_radar_ref=1`, `nwp_diagnostics=0` | `diagnostics.morrison_radar` (`refl10cm_hm`, `radar_init` constants in WRF literal kinds); MP-exit columns, full column at history alarms, `MAX(-35,dBZ)`; mp40 MSKF = REFL (MSKF cumulus arrays unallocated under `aercu_opt=0`, `UPSTREAM_WRF_BUGS.md`) | Pristine REAL4 column oracle (`morr_radar_oracle/`, both module copies, 304 columns incl. 150 melting-layer): 13376/13376 BYTE. Original CPU-WRF c08 rerun with `do_radar_ref=1` from a private alias-free stage (`C08R_CPU_RUN.json`, `C08R_STAGE.json`, `stage_c08_radar_cpu.py`; 25/25 wrfout byte-identical to the superseded first rerun), recompute from history over 24 frames (`MORR_REFL_CPU_HISTORY_c08.json`): 19 frames <= 0.002 dBZ. 84 column-frames in hours 10-14 differ up to 15.45 dBZ; attribution HYPOTHESIS: history does not hold the exact MP-exit state (`T1D` rebuilt from TH/P). Proven only: the pristine REAL4 oracle on those reconstructed operands equals the port BYTE (`MORR_REFL_OUTLIERS_ORACLE.json`); not a coupled-trajectory pass |

Release identity on the final trees: Swiss `f672a573`, PROD `6fc715a2` (`RELEASE_IDENTITY_*_cd9d.json`, `*_6eba.json`, rebased `*_03a7.json`).

Critic round rv-t-classd (FIX on 165de18b3): F1 native checkpoints now bind the BEP/BEM selection and
domain and validate `QGR_URB3D` on save/readback/public read; F2 the Morrison radar constants (and the
NSSL radar log10 split constants) enter `baked_table_hash`/AOT keys; F3 the NSSL REAL `Log10` is a
REAL-only float-float evaluation (fp32 still BYTE vs pristine S6, no f64 log); F4 the c08 rerun was
staged privately: an earlier `cp -P` copy reproduced v-prep's `namelist.input` alias and the radar line
was written through it into the sealed reference (restored by its owner 17:30Z, incident preserved
there). `C08R_CPU_RUN.json` now records original (sealed) and used SHA-256 separately.

Finding (not fixed here, reported to the manager 2026-10-15T12:26Z): WRF passes the DRY density
`1/(al+alb)` to microphysics (`moist_physics_prep_em`, module_big_step_utilities_em.F:5542). The port's
NSSL, Ferrier and Goddard 4-ice adapters pass `(1+qv)/alt`; the c14/c15 recompute shows 0.043 dBZ
systematic for that convention against 7.6e-6 dBZ for the dry density.

Scripts: `nssl_refl_cpu_history.py`, `urban_cpu_all_frames.py`, `constant_cpu_all_frames.py`,
`morr_radar_oracle/{make_inputs.py,build_run.sh,freeze.py}`, `morr_refl_cpu_history.py`,
`morr_refl_outliers_oracle.py`, `morr_refl_outliers_t_sensitivity.py`.
Tests: `tests/v040/test_nssl_one_moment_radar.py`, `test_classd_urban_green_roof.py`,
`test_classd_constant_fields.py`, `test_classd_morrison_radar.py`.
