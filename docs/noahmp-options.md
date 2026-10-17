# Noah-MP options in v0.4 development

The native `sf_surface_physics=4` path admits the choices below. Each new choice has a `PhysicsStepSpec`, pristine WRF 4.7.1 REAL4 source oracle, operational consumer and guarded restart. Qualification is at component/caller boundaries; GPU execution, whole-executable resource/E41 measurements and coupled forecast D6 against CPU-WRF remain pending. Deviceless native component prescreens and short CPU integration/restart smokes are separate evidence. Small compiled deviations are disclosed in the qualification receipts and `KNOWN_ISSUES.md`; literal operation-order gates match the pristine references byte for byte.

| Control | Admitted choices | Behavior |
|---|---|---|
| `dveg` | 1–9 | Prescribed, calculated or maximum vegetation fractions; monthly/input LAI or dynamic carbon according to WRF |
| `opt_crs`, `opt_btr` | 1/2; 1/2/3 | Ball–Berry/Jarvis; Noah/CLM/SSiB moisture stress |
| `opt_run` | 1–8 | SIMGM groundwater, equilibrium table, original, BATS, MMF groundwater, VIC, Xinanjiang, dynamic VIC |
| `opt_infdv` | 0/1/2/3 | Inactive 0; Philip, Green–Ampt, Smith–Parlange for run=8 |
| `opt_sfc`, `opt_frz`, `opt_inf` | 1/2 each | Original WRF drag, supercooled-water and frozen-permeability branches |
| `opt_rad` | 1/2/3 | Original modified/grid-cell/vegetated-fraction two-stream geometry |
| `opt_alb`, `opt_snf` | 1/2; 1–5 | BATS/CLASS snow albedo; all five rain/snow partition methods |
| `opt_tbot`, `opt_stc` | 1/2; 1/2/3 | Zero-flux/prescribed bottom and original thermal coupling choices |
| `opt_rsf` | 1–4 | Sakaguchi–Zeng, Sellers, adjusted and harmonic snow resistance |
| `opt_soil`, `opt_pedo` | 1/2/3; 1 | Surface categories, four-layer SOILCL maps, composition with Saxton–Rawls |
| `opt_crop` | 0/1 | Disabled/Liu crop; crop fractions, calendar and season-GDD maps, crop photosynthesis rows |
| `opt_irr`, `opt_irrm` | 0–3 each | Disabled or WRF triggers; mapped/sprinkler/micro/flood methods |
| `opt_tdrn` | 0/1/2 | Disabled/simple/Hooghoudt drainage, with original layer selection and map thresholds |
| `opt_gla` | 1/2 | Approved independent glacier timestep from s2-glacier; original-source and restart qualification inherited |
| `sst_skin` | 0/1 | WRF ocean skin phase with persistent bulk SST, SSTSK and DTW |

Dynamic VIC is **experimental**: pristine WRF reads unassigned `YD`. The binding compiler-zero policy is mirrored at those sites. Changing only compiler initialization changed 52/480 reference outputs, maximum 0.0026633055; see [upstream defects](../proofs/u2_land/UPSTREAM_WRF_BUGS.md). Run=8 requires explicit `opt_infdv=1/2/3`; WRF leaves `FSUR` unassigned for 0. Tile drainage runs only with run=3, as in WRF. Its equivalent-depth calculation uses static table parameters and is computed once during binding.

Dynamic carbon and Liu crop require Ball–Berry (`opt_crs=1`); Jarvis returns WRF's inactive-carbon PSN sentinel. Their compact lake/urban combinations are refused. Irrigation, prognostic groundwater and tile drainage with compact lake tiles are refused. SST skin requires RRTMG 4/4 and ice-free prescribed SST. CPU-history replay does not execute these nondefault controls.

Liu crop follows original cold initialization and strict corn/soy fraction selection, generic tie fallback, INTEGER calendar assignment and `SEASON_GDD/1770` scaling. Phenology reads the previous growth stage before ENERGY; current ENERGY T2 enters GDD after WATER. Soil composition overrides occur during the driver call; cold SH2O initialization intentionally still uses surface ISLTYP, exactly as WRF.

New carries are opt-in: irrigation counters/remaining event amounts/accumulators; six carbon pools, or eight pools plus INTEGER PGS for Liu; positive-depth groundwater ZWT/WA/WT; one QTDRAIN accumulator; and three SST-skin fields. Inactive selections retain the released program and carry prefix. Runoff/tile fine substeps share one optional native column phase; there is no host transfer in stepping. GPU cost remains unmeasured.

`opt_run=5` is provided by main's approved MMF port, with official groundwater geography, one ice-free domain and its qualified Noah-MP defaults except opt_inf1/2; combining it with the other nondefault controls remains refused. `opt_gla=1/2` use the approved independent glacier model; glacier domains refuse opt_snf=5 because WRF defines only branches1–4 there. GECROS crop=2 remains refused with the upstream-broken reason from the main catalog. Soil=4 has no defined WRF 4.7.1 branch. WRF implements surface mosaic only for Noah-classic (sf=2); it ignores it with Noah-MP. Noah-classic is API-wired/admitted but remains absent from native CLI input binding. Its separate mosaic driver is an option-specific infrastructure blocker: per-tile category/REDPRM binding, evolving surface and soil states, and their restart schema are absent; see the source audit in proofs/u2_land/OPTION_BLOCKERS.md.
