# Support matrix — v0.4.4

**27 October 2026. RW3 PASS by the Owner-approved manager decision, with the two disclosed exceptions below.**
This matrix distinguishes implemented options, measured qualification and
refused capabilities. A component PASS is not three-nest, station or long-run
qualification. [Release notes](../../release_notes/RELEASE_NOTES_v0.4.4.md)
contain the full gate and exception table.

## Default and release gates

| Item | Status | Evidence / limits |
|---|---|---|
| Default physics availability | Existing default suite: Thompson/8, MYNN/5, Noah-MP/4, RRTMG/4 and configured cumulus | Availability is distinct from individual qualification scopes |
| Nested terrain shading | Included correction; intentional result change with slope_rad=1 and topo_shading=1 | Original-WRF mask/clock evidence; composed RW3 passes with the disclosed floor/trace exceptions |
| MYNN QKE/EL/QSQ fresh-start seed | Included | Original-WRF initialization/oracle scope; QKE snow handling and EL/variance retention change |
| Solar/ozone division | Included, 14 protected quotient families | Nine solar/five ozone; remaining families and representation residuals explicitly unqualified |
| Host refusal/reader-registry/version changes | Included | Default numerical operators unchanged by the host policy/metadata deltas; package/cache/checkpoint identities can change |
| CPU suite | 4209 PASS /34 SKIP, release defaults102 PASS /1XFAIL | No unresolved source failure after the recorded classification; two absent W4 paths N_A, not PASS |
| Warm replay RW1 | PASS, recorded short cases | Zero new compilation/byte-identical paired outputs; regression/cache evidence |
| Cross-date RW2 / bundle replay RW4 | PASS, recorded dates | Same-grid cache keys and sealed-bundle replays; no new weather-skill claim |
| Original CPU-WRF RW3,72h | **PASS by approved decision** |219 frames; d01/d02 PASS; d03 RAINNC67–72 six FLOOR-LIMITED checks,0 REAL FAIL; final QGRAUP t72 is the registered v0.4.2-class exception |
| Same-session release timing | PASS, separate from fidelity | Three24h pairs: step5.91100N/5.92033O s/fc-h; ratio 0.9984 (0.998 rounded), spread0.0279; no resolved speed change |
| Same-session72h old version | NOT RUN | O72 skipped for disk; no 72h N/O pair claimed |

Default program identities differ from v0.4.3 for all five canonical compiler-input
pairs. See [identity table](../../proofs/w44_rc/IDENTITY_TABLE.json),
[CPU gate](../../proofs/w44_rc/FINAL_GATE_TABLE.json),
[known issues](../../proofs/w44_rc/KNOWN_ISSUES_RC.md) and
[release facts](../../proofs/w44_docs/receipts/V044_RELEASE_FACTS.json).

## Regional experiment tools

| Tool / setting | v0.4.4 status | Qualification boundary |
|---|---|---|
| Existing Track-P surface controls, including z0mvt | Available, experimental | Swiss GPU24h and changed-table response evidence at its recorded scope; not all regions/parameters jointly qualified |
| MYNN runtime coefficients: sqfac, alp1, cphm_unst, am_st, entrainment | **Experimental** | Actual PROD two-domain GPU component/root PASS, default BYTE, same keys/no new native blobs, resource checks PASS; no three-nest D6, station or24–72h drift qualification |
| Thompson runtime nt_c, tno, ato, av_c | Experimental, component/short-root PASS | Microphysics, not PBL mixing; no full three-nest/D6/station qualification |
| Thompson av_i, av_r, autoconversion extensions | Refused | CPU research evidence does not qualify the unsupported GPU family |
| Diffusion/damping runtime overrides, including c_s | **Not shipped; refused** | D1 GPU STACK288>256 and ON-default!=OFF101 leaves. May come later after GPU qualification; no date or commitment |
| New LSM/Jarvis smcmax,dksat,bexp,psisat,hvt,hvb,z0sno,snow_emis,rsmin | **Not shipped; refused** | All nine blocked by RC item9; older Track-P controls remain separate |
| LANDUSE dataset admission | MODIFIED_IGBP_MODIS_NOAH required for Noah-MP | USGS/missing/unknown labels refuse; relabeling is not conversion, and no generic table-scaling qualification is implied |
| Input overrides | Host/file scope accepted; edited forecasts experimental | Private copied files, strict schema/finite/physical admission; ZNT/EMISS/ALBBCK storage edits have no Noah-MP effect |
| Atmospheric warm start | Experimental | File/CPU/original-WRF initialization evidence; older completed reference and regenerated-input t0 must remain distinguished; no full paired GPU forecast qualification |
| MYNN budgets / EDMF-off | Experimental | Original-WRF/component evidence; coupled GPU ON/fulllead qualification remains incomplete, not upgraded by default RW replays |
| rad_diag=1 / QI_BL / icloud_bl | Scheme-specific diagnostic support as documented | RRTMG constraints; recorded short research run/native/cache scope is not full release/observation qualification |
| Checkpoint branches / short windows | Supported admission with experimental edited forecast scope | Parameter signatures/digests fixed; cannot enable refused families or change static physics silently |
| Precompile variant queue / ordinary GPU priming | Supported | CPU planning, standard GPU execution and artifact completion checks; no universal readiness guarantee |
| Deviceless compile -> GPU load | **Unavailable** | Admissions refused incompatible fingerprints; W4 shelved, W5 remains |
| Offline perturb / ensemble planning | Implemented CPU tools | Coherent unbalanced fractional wind IC/BC members; not calibrated spread or balanced analysis |
| Stochastic physics | Experimental; Swiss d01 GPU24h scope | SKEBS/SPPT/SPP pattern+seed/original-WRF evidence; no general three-nest stochastic qualification |

Tool contracts: [regional guide](../regional_optimization.md),
[surface controls](../surface_params.md), [MYNN family](../runtime-parameters.md),
[Thompson](../thompson-parameters.md), [overrides](../initial_overrides.md),
[warm start](../atmospheric-warm-start.md), [budgets](../mynn-budgets.md),
[radiation](../radiation_diagnostics.md), [precompile](../precompile.md),
[ensembles](../ensemble.md), [stochastic](../stochastic.md).

## Existing physics options

The [physics menu](../physics.html), [namelist contract](../namelist-compatibility.md)
and [v0.4.3 option matrix](SUPPORT_MATRIX_v0.4.3.md) retain their implementation
and historical per-configuration evidence. Those earlier24h/72h tests identify
what was measured then; this matrix does not relabel every old option combination
as freshly qualified on the changed v0.4.4 program. Existing hard refusals and
required scheme pairings remain unless a row above says otherwise.

## Continuing limits

Root-walk, constant-folding and command-buffer default flips are OUT. The ring-timing
lead and PROD d02 graph-partition limit remain documented. The component-only664
fusion must not be labelled a new canonical default regression or an unmeasured
zero-cost kernel. E84 leaves13 ranked families plus evolving physical divisions
outside its proof. Internal low-level fingerprint opt-outs/legacy metadata remain
a hardening item, not an end-user deviceless GPU route. Existing long-run heavy-rain,
nine-domain and teacher-integrity issues remain in [KNOWN_ISSUES](../../KNOWN_ISSUES.md).
No station-skill, full-process VRAM or global device-operation claim follows from
the short component/root results.


## Disclosed RW3 exceptions

At d03lead67 RAINNC raw RMSE rose from 0.50 mm in v0.4.3 to 1.06 mm. The
same original CPU-pair archived metric gives C67=0.8905983 and C/X=0.840143,
meeting the unchanged 75% floor rule. Leads67–72 remain raw D6 exceedances;
the binding result is six FLOOR-LIMITED checks, not a raw-threshold pass.
QGRAUP t72 is GPU0 versus CPUmax6.5775e-12 kg/kg, explicitly accepted as the
same registered v0.4.2 exception class. See the
[approved decision](../../proofs/w44_docs/receipts/MANAGER_DECISION_RW3_20261027.md).

O72 was not run because of disk admission; speed and memory use three
same-session 24h pairs only. Rounded N/O ratios are 0.998 speed,0.996 VRAM
and 0.992 host RSS. No same-session 72h old-release comparison is claimed.
The original raw FAIL report is retained alongside the approved disposition.
