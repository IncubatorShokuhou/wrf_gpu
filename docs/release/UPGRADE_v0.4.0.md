# Upgrading to v0.4.0

**Released v0.4.0 — defaults frozen; all66 registered scientific scopes pass.**
The00:26Z final checkpoint is66/66 accepted-scope scientific D6 PASS (65 full24h + c45 accepted6h); see the
[registered-source table](FINAL_COMPOSITES_v0.4.0.md).
Use the [support matrix](SUPPORT_MATRIX_v0.4.0.md) and its
[frozen evidence](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json) to check the
specific scheme combination, domain layout and prepared inputs you intend to
run. A catalog entry means a wired path, not universal forecast qualification.

## Start affected forecasts again

Older non-default runs may contain incorrect precipitation totals,
thermodynamic handoffs or untransported number species. Check the
[affected scheme/version list](../../KNOWN_ISSUES.md#non-thompson-precipitation-and-wdm-caller-defects).
Old accumulated precipitation is preserved on restart; it is not migrated to
correct totals. Start a fresh forecast to obtain corrected accumulations.
Runs from v0.3.3 and earlier that requested non-default physics may instead
have executed the default suite; see the existing
[namelist-binding disclosure](../../release_notes/RELEASE_NOTES_v0.3.4.md).

Glacier points now use the separate Noah-MP glacier timestep. The Swiss default
program intentionally changes; the previous Swiss program identity cannot
justify carrying its glacier validation forward. WN3 and other ice-free
geographies require their own program/input identity certificates. Evidence:
[glacier issue](../../KNOWN_ISSUES.md#noah-mp-glacier-points-in-shipped-versions-through-v034),
[c00/WN3 receipts](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json).

## Checkpoint and restart compatibility

Keep original inputs and use fresh output/checkpoint directories for a new
forecast. Do not edit authenticated checkpoints to bypass schema errors.
The CLI's full-carry checkpoint generations and the NetCDF/legacy State-only
APIs are different formats; compatibility of one does not prove compatibility
of the others. [Existing schema disclosure](../../KNOWN_ISSUES.md#v034-restart-schema-gaps).

Active schemes now require their additional prognostic numbers, held physics
state and diagnostics. Depending on the selected configuration, these include
CAM-UW/QNSE/TEMF/epsilon-PBL carries, CAM radiation state, NSSL/P3/NTU/CAM-MG
species, glacier snow forcing, urban/lake/ocean state and nudging/feature
cursors. An old file missing required active state is refused under the E78
checks; an absent inactive optional field can be accepted only by that
format's explicit compatibility rule. There is no blanket cross-version
restart guarantee. [Integration receipts](../../proofs/s2_integrate/batch15/QUALIFICATION.json),
[per-scheme evidence](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json).

P3/Jensen held size/density/fall-speed/aspect-ratio diagnostics must retain
their original time level; active payloads missing required arrays are refused.
Newer radar retention is independently qualified and does not retroactively
complete old GPU history inventories. Evidence:
[ice diagnostic schema](../physics/ice_diagnostics.md),
[radar retention receipt](../../proofs/t_writer_p3_radar/QUALIFICATION.json).

For a legacy CAM-UW NetCDF file without saved PBL selection, supply the
resolved inactive namelist to `read_wrfrst_carry(path, namelist=...)` only when
CAM-UW is actually inactive. Regenerate active checkpoints that predate the
group/schema marker. [CAM-UW compatibility rules](../../KNOWN_ISSUES.md#v034-restart-schema-gaps).

## New selections and inputs

| Keys / workflow | Upgrade action and scope | Evidence |
| --- | --- | --- |
| `mp_physics`, `cu_physics`, `bl_pbl_physics`, `sf_sfclay_physics`, `ra_lw_physics`, `ra_sw_physics` | CAM-MG11 uses the merged stock-Triton3.6.x route; install its pinned optional dependency and retain its E41/final-qualification disclosures. Select only an admitted combination. QNSE is PBL/SFCLAY 4/4; TEMF is 10/10. NSSL 17 normalizes to 18 under WRF 4.7.1. FLG/GFDL are CPU-only. | [Per-code matrix](SUPPORT_MATRIX_v0.4.0.md) |
| `sf_surface_physics=3/8` | Prepare RUC six/nine-layer or SSiB three-layer inputs with original WPS/real.exe; do not relabel Noah soil arrays. | [RUC](../../proofs/s2_rucfix/README.md), [SSiB](../../proofs/s2_ssib/README.md) |
| `diff_opt=2`, `km_opt=2/5` | LES/SMS use separate TKE state and restricted PBL/surface/scalar combinations. `km_opt=3` is GPU-refused after a nonfinite Swiss run and failed CPU references. | [SMS scope](../../proofs/s1_km5/QUALIFICATION.md), [matrix](SUPPORT_MATRIX_v0.4.0.md) |
| `grid_fdda`, `obs_nudge_opt`, `grid_sfdda` | Supply the required analysis/observation streams and obey domain/pairing restrictions. The final source5c15881ede c77/c78 receipts have full24h D6/coverage and E41 PASS; see the final composite ledger. | [Surface nudging](../surface_nudging.md), [admission snapshot](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json) |
| `dfi_opt=1/2/3` | Use supported single-domain windows/filter/land settings; DFI with nesting or unsupported forcing remains refused. Natural DFL output times must be retained. | [DFI evidence](../../proofs/s1_dfi12/), [matrix](SUPPORT_MATRIX_v0.4.0.md) |
| Noah-MP `dveg`, `opt_crop`, `opt_soil`, `opt_irr`, `opt_irrm`, `opt_run`, `sst_skin` | Supply geographically valid WRF/NCAR maps for active crop/irrigation/groundwater branches. `sf_surface_mosaic=1` is refused. Column tests do not qualify absent maps. | [Land qualification](../../proofs/u2_land/QUALIFICATION.json), [MMF inputs](../../proofs/s2_mmf/INPUT_REQUIREMENTS.md) |
| `sf_urban_physics`, `use_wudapt_lcz`, `sf_lake_physics`, `sf_ocean_physics`, `isftcflx` | Prepare compatible categories, morphology and water/soil inputs. Check uncomputed BEM green-roof/PV fields. | [Urban/lake/ocean scope](SUPPORT_MATRIX_v0.4.0.md#other-options-and-workflows) |
| `nwp_diagnostics`, `prec_acc_dt`, `do_radar_ref`, tslist / regional streams | Declare sampling and accumulation windows. Instantaneous products are not sub-hourly extrema. Feature-specific resume restrictions apply. | [Online diagnostics](../../proofs/u2_diag/QUALIFICATION.json), [streams](../../proofs/u2_streams/QUALIFICATION.json) |
| `skebs`, `sppt`, `spp*`, `rand_perturb`, `windfarm_opt=1`, `tracer_opt=2` | Supply stochastic/turbine/tracer configuration. Custom tracer inputs use `gpuwrf_tracer_file`; WRF-Chem modes and two-way tracer feedback are refused. | [Stochastic](../../proofs/u2_stoch/QUALIFICATION.md), [turbines](../wind-farms.md), [tracers](../../proofs/u2_tracers/QUALIFICATION.json) |
| `feedback=1`, `smooth_option` | Two-way nesting is opt-in; the source-specific full two-way comparison does not establish arbitrary configurations or release-default one-way fidelity. | [Two-way qualification](../../proofs/u2_twoway/QUALIFICATION.json), [current rows](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json) |

Namelist preflight remains authoritative for exact values and pairings.
Options set away from an unbound WRF default fail loudly. Do not silence a
refusal by dropping a scientifically required option.
[Binding rules](../namelist-compatibility.md#cli-binding-v034).

For advanced options with only single-domain evidence, explicitly retain
**nested not validated** where nested admission is allowed. Actual refused
pairings remain refused. A preflight success is not a nested scientific gate.
Frozen release identities on5c15881ede are Swiss `3356e24924a9207b2264cec1dc8df0591cd85e1a856097251b6f2efeb7327c57`
and PROD d01 `196aed9f9973499c5610fdf9965223503a61944156b8cd41357fd8c6e911911e`.
F20/C5′ default D6/E41 gates are closed. Old f672a573/6fc715a2 and2bf12cb8
remain historical. Program-changing caller repairs require fresh forecasts for
corrected accumulated precipitation and authenticated active checkpoint schemas.
[Default gate and source/identity proof](../../proofs/s2_integrate/batch62/gate_WN3_closed/GATE_CLOSURE.json),
[fix inventory](FIXES_v0.4.0.md).

## Caches, resources and verification

Cache/program keys include selected physics constants and configuration.
Expect a cold setup when identity changes; do not force old cache keys or
infer byte-exact output from identical source alone. Actual compile/readiness,
peak VRAM and warm S1 for the released workload remain separate measurements.
[Source/cache qualification record](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json).

CPU test memory varies substantially. Use the measured heavy CI tier rather
than treating a low-cap OOM as a numerical failure or skipping the tests.
The SMS pristine test used 12.525 GiB peak RSS and the seven-nest terminal
test 17.043 GiB [M]; each passed under a 24 GiB/no-swap cap. Several PROD
fixtures also require the heavy tier. These are test-harness observations,
not forecast memory requirements. The exact test/attempt/source list is
bound as `heavy_tests` in the [evidence snapshot](../../proofs/r_docs040/EVIDENCE_SNAPSHOT.json).

Before adopting a new configuration, review its D6 coverage, REPORT_ONLY
fields, E41 and F2 disclosures, then evaluate the variables and station skill
your application needs. Final accepted-scope scientific validation is complete; source and tags are published, while GitHub release assets await owner authentication.

## Diagnostic producers and resource disclosures

Class-D initialized/held/radar producers are merged. Active green-roof/PV,
black-carbon and aerosol-optics branches remain refused in the stated scopes.
Jensen55 now retains radar every MP call, NTU56 writes prognostic QHAIL, and
CAM-MG11 writes WRF's initialized DLF2 zero only with shallow convection off.
Older incomplete outputs stay incomplete; a schema/CPU producer check does
not qualify a new GPU trajectory or cost. Follow the
[producer scope](../../KNOWN_ISSUES.md#v040-history-fields-without-a-producer)
and [source-bound reconciliation](../../proofs/r_docs040/RECONCILIATION.json).

Default WN3/PROD/Swiss F20 programs must pass E41. Opt-in composites require
strict original-CPU D6, full inventory and NF0; resource failures retain raw
E41 FAIL and are disclosed per executable. No E41 threshold or D6 tolerance
changes. F2 event justification remains separate.
