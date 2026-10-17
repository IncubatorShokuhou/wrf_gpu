# F2 guard-growth successor — t-camuw

Producer disposition: the **guard-count growth is real**, and the retained four-hour evidence justifies it as finite, tiny mass residues on the WRF specified boundary. This is **not** the standing same-count/control exception. No positivity clamp or scientific tolerance changed. Full GPU qualification remains separately scheduled.

## Same work and complete readout

Corrected source `702096abf52063eebe0645d74a23ef3015733d8b` and no-fix control `5098e87a981d5b591704239dbfee873281ea839b` use identical input SHA256s, CPU backend, 800 steps, dt=18 s, BL9/MP8/SF5/Noah-MP4/RRTMG4/CU0, bldt=0, FAST1/CENSUS1/X64ON and strict guards. The control reverts only CAM raw-RK coupling, hydrostatic P/P8W and resolved-cloud selection; exact deltas are retained. It is a regression-attribution control, not a WRF reference.

All 60 phase/field rows are retained in `FINAL_FORECAST_CENSUS.json`: 48 REAL32 rows observed on every one of 800 steps, 12 explicitly absent (Nh/nwfa/nifa). Ns/Ng/Nc/Nn are present zero arrays. All phase nonfinite and above-cap counts are zero. Guard repairs are disabled and their signed water budgets are zero. Existing WRF physics cleanup is unchanged.

The 81,424 aggregate equals 40,712 negative mass visits counted twice by the `out_of_range` and `repaired` predicate columns. It is neither a unique-cell count nor executed repairs. The matched control is 1,542 aggregate / 771 negative mass visits. Old GPU 24h 7,948 was not a matched comparison.

## Per-field extrema and locations

Below: cumulative `pre_dynamics_guard` negative visits. All visits in both arms are **ring 0**, with zero interior visits. Ring 0 is j=0/41 or i=0/41 at every k. Values are kg/kg for mass and number/kg for Ni/Nr. Minimum syntax is value; step; (k,j,i). Other fields and all other legacy guard predicates are zero. Ni/Nr are observed by F2 but are not included in the legacy guard aggregate.

| Field | Corrected visits | Control visits | Corrected minimum | Control minimum |
|---|---:|---:|---|---|
| qc | 29,208 | 333 | -2.002136436e-15; 792; (2, 8, 0) | -2.907969818e-13; 605; (16, 36, 41) |
| qr | 2,937 | 261 | -6.161724735e-19; 647; (0, 41, 24) | -4.450060469e-18; 540; (14, 40, 41) |
| qi | 4,533 | 56 | -3.240590561e-17; 236; (25, 0, 32) | -2.956592956e-19; 224; (28, 0, 37) |
| qs | 3,044 | 116 | -1.428019923e-19; 686; (14, 41, 14) | -4.736221948e-18; 573; (15, 16, 41) |
| qg | 990 | 5 | -5.185368302e-19; 695; (8, 41, 30) | -1.679619444e-19; 496; (16, 36, 41) |
| Ni | 16,725 | 61 | -1.420829562e-06; 254; (25, 0, 33) | -2.511153070e-05; 707; (19, 41, 34) |
| Nr | 11,570 | 178 | -1.725515841e-10; 613; (8, 41, 29) | -1.512422743e-09; 564; (14, 32, 41) |

Full counts, first samples, minima/steps/indices and ring/interior splits for **every phase** are retained, including the larger interior post-RK negatives in both arms. Those pre-microphysics excursions are **not classified wholesale as round-off**. This successor supplies the missing guard-growth evidence, not a blanket waiver of the separate full GPU F2 gate.

## Pristine-WRF justification

The read-only original-WRF census (`ORIGINAL_WRF_NEGATIVES.json`, five source-file SHA256s) observes h0–4. Across h1–4, original WRF has **422 negative endpoints** (246 mass + 176 number), all on ring 0; corrected has **373** (198 + 175), while the no-fix control has **4** (3 + 1). Census endpoint differences were independently checked against all 48 retained mass-array snapshots. Hourly endpoints do not assert internal-stage equality with WRF.

Original WRF cloud-water h4 minimum is −1.571802724e−15 kg/kg at **(2,8,0)**, the same cell as the corrected all-step minimum −2.002136436e−15. There are zero corrected interior guard candidates; the increase is repeated boundary-residue visits, not a new interior guard failure.

Pristine source explains the boundary policy: CAM `update_phy_ten` uses `add_a2a` (`phys/module_physics_addtendc.F:903–969`), whose specified/nested bounds skip ring 0 (`:2435–2481`). WRF applies the coupled scalar source before final-stage advection (`dyn_em/solve_em.F:1932–2000`, `module_em.F:1803–1916`); this update has no positivity clamp. WRF microphysics skips the specified zone (`phys/module_microphysics_driver.F:809–813,879–886`). Tiny boundary negatives therefore remain observable. The old CAM route instead supplied whole-field post-RK Euler increments and omitted the CAM scalar RK sources; its near-zero endpoint census is not a fidelity target. Restoring the WRF source route changes rounding and boundary-residue persistence. Attribution of an exact fraction of the count delta to each of the three fixes is not claimed.

During this audit, a separate CAM-QNI caller gap was corrected: MYNN writes a full-tile number source, whereas CAM QNI goes through `add_a2a`. CAM now reuses the existing boundary-ownership helper for Ni. Ten targeted tests pass; deletion gives one numerical AssertionError and zero setup errors. This hardening **does not remove the Swiss guard growth**: final full census and all 16 retained field/hour arrays are exactly the prior corrected result. It changes no persistent carry or TPERT/KPBL interface.

## Validation and cost

Final T2 RMSE [M CPU vs original WRF] h1–4: **0.085830943 / 0.087998740 / 0.093638507 / 0.083913682 K**, all <1.5 K. Original cloud/CAM column qualification and completed shared tests are preserved; no unchanged oracle rerun. Rebase proof matches the selected graph f2f24c16…, all 61 constants and all 187 initial carry arrays, so the completed forecast/census transfers to the rebased source.

Final CPU collection wall [M] 1483.429994 s, peak RSS 9,641,868 KiB; control 1031.051375 s, 9,098,572 KiB. These are unpaired shared-CPU runs, not a speed claim. The CAM-only Ni ownership operation constructs one field-sized source result (310,464 bytes on Swiss [I]); no extra CAM solve, persistent carry, host transfer or synchronization is introduced by that source hunk. Compiled GPU launch/temporary/readiness/wall costs remain [U]. Final rebased native release graph [M] is exactly 2bf12cb8374d3a84a440e47dea4b77b0c7436108fc2ee2616bba6ec79b17962a (load 34.5 s, trace 17.6 s, wall 52.6 s). Final rebased CAM/cloud tests: 18 PASS, 39.55 s, zero failures/errors/skips.

Original sealed `proofs/t_camuw/receipts/` is unchanged. This directory is its successor. Raw census NPZs and retained hourly fields remain under `<USER_HOME>/wrf_gpu2_lanes/t-camuw/`; disposable caches and snapshots are removed after their last dependent arm.
