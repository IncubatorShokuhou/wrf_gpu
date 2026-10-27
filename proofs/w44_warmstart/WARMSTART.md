# Atmospheric warm initialization

Candidate under qualification. The same native files completed a three-nest,
three-hour CPU-WRF cycling twin using the approved package-guarded Gen2 build.
Actual GPU initialization, GPU forecast and paired D6 gates remain open.

`gpuwrf cycle` stays land-only by default. Native atmospheric cycling is explicit:

```sh
gpuwrf cycle --atmosphere --native-atmosphere \
  --source previous/wrfout_d01_VALID_TIME \
  --wrfinput fresh/wrfinput_d01 --output cycled/wrfinput_d01 \
  --wrfbdy fresh/wrfbdy_d01 --boundary-output cycled/wrfbdy_d01 \
  --nest previous/wrfout_d02_VALID_TIME fresh/wrfinput_d02 cycled/wrfinput_d02 \
  --nest previous/wrfout_d03_VALID_TIME fresh/wrfinput_d03 cycled/wrfinput_d03
```

Use every source at the fresh input's exact valid time and run with
`&time_control cycling=.true.`. Supply fresh `wrflowinp_dNN` alongside the emitted
inputs and boundary. Native mode retains the source's resolved terrain and base
state. WRF history/restart writes `SAVE_TOPO_FROM_REAL=0`; this prevents a second
nest terrain blend. See output_wrf.F
1004–1021 and mediation_integrate.F
739. Raw real.exe inputs carry 1. The measured May pair has child terrain
changes of 344.9 m and 69.3 m between resolved source and raw fresh inputs.

Grid coordinates, map factors, vertical coordinates, categories, schemes, nest
relationships and valid time must agree. Unsupported inputs refuse. All files
are staged and validated before publication; existing outputs are never replaced.
A crash during multi-file publication can leave a partial set: consume only a
successful invocation whose JSON receipt lists every output.

| Field | Native transfer |
|---|---|
| U, V, W, T, THM, P, PH, MU | Exact source storage, with native staggering and theta conventions |
| Active moisture and number species | Registry-backed field map; required missing species refuse |
| QKE and recorded MYNN state | Exact recorded state; lowercase qke_adv supported; absent history qke_adv initialized from source QKE with provenance |
| Land/snow | Existing land admission, source masks and per-field hashes; soil, CANWAT and land snow stay cycled (WRFDA `update_lsm=.false.`, `keep_snow_wrf=.true.`) |
| Lower boundary: TMN, SST, VEGFRA, ALBBCK, SEAICE (+ admitted-equal LANDMASK, XLAND, IVGTYP, ISLTYP) | Fresh real.exe input, as WRFDA `update_low_bdy` (default `.true.` when cycling, always for nests): da_update_bc.f90 96–110, 409–436. A field present in the cycled file but missing from the fresh input refuses |
| TSK | `keep_tsk_wrf=.true.`: cycled where the fresh IVGTYP is not water, fresh over water (365–405); ISWATER is the fresh input's category attribute |
| SNOW, SNOWC, SNOWH over water | Zeroed where IVGTYP is water and SNOW > 0 (476–517) |
| HGT, PB, PHB, MUB | Source resolved values |
| Other recorded source variables | Single source frame, exact hashes; issue-clock exceptions reported |
| Missing input-only slots | Fresh real.exe, each slot reported |
| Missing ALB/T_INIT | Existing WRF reference-profile calculation using SOURCE PB and profile parameters |
| Missing AL | Source PH/MU inverse density minus source-derived ALB |
| START_DATE/SIMULATION_START_DATE/GMT/JULYR/JULDAY, STEP_NUMBER | Fresh issue metadata/step, explicitly reported |
| XTIME | Fresh value when present in real.exe input; otherwise original source diagnostic/units retained. Runtime elapsed time restarts at zero, as WRF start_em does |
| Boundary and lower forcing | New real.exe forcing; first root boundary interval adjusted as below |

The optional `--atmosphere` workflow without `--native-atmosphere` overlays fresh
bases. It is limited to d01 and roundoff-sized base/terrain differences. It uses
`new = old + old_base - fresh_base` with float64 host intermediates and original
storage dtype; equal bases preserve bytes. Material differences refuse. No
terrain translation or blending is implemented.

## Admission and initialization

The shared `validate_wrfinput(..., transport_reference=...)` implementation was
merged on main at c15f937487. Unchanged transported fields require exact source
storage/totals identity, finite data, schema and geometry. The receipt reports
inherited extrema/counts, including negative species and supersaturation.
Actual user edits retain strict positivity, RH and hydrostatic checks. No clamps
or silent waivers are used. The May original source contains QNRAIN
-6.1868e-29 in two outer-ring cells and liquid RH up to 1.0049; these are reported.

File preparation supports the existing Registry species map. Runtime cycling is
currently restricted to Thompson MP8, MYNN5, DFI0, HYPSOMETRIC_OPT2 and
USE_THETA_M0/1. Conflicting restart aliases and unmarked cycling inputs refuse.
This is warm initialization: integrator scratch and unavailable state are not
an exact restart. NPZ/NetCDF restart markers distinguish cycling from cold mode.

start_em.F 767–777 invokes
`rebalance_cycl` for cycling/DFI0; 2259–2544 rebuild pressure/geopotential without
RH adjustment. The host port uses the existing scalar WRF REAL32 math shim.
Native child initialization skips the second terrain blend, retains prognostic
THM, then follows adjust_tempqv, cycling rebalance and press_adj. W follows the
WRF `use_input_w` gate; the default rebuilds it even if the carried W is nonzero.

The MYNN driver 618–752
selects INITIALIZE_QKE from the incoming per-column maximum (<REAL(0.0002)),
then clears QKE, moments and BL cloud for non-restart cycling. Thus high-QKE
columns start with zero QKE. `mym_initialize` recomputes EL in both branches and
QSQ in the cold branch. The optional helper extension passes the 40-case original witness in compiled
CPU AVX mode, which prevents FMA contraction like the reference gfortran build:
QKE maximum error 9.54e-7, QSQ 2.84e-14, EL 2.29e-5 m. Generic CPU FMA changes
a near-neutral virtual-temperature gradient and fails the fixed QSQ bound
(ratio 4.766); barriers did not help and were discarded. The actual GPU oracle
must pass those unchanged bounds before any GPU forecast. There is no keep-QKE extension.

The original WRF 4.7.1 cycling bucket loop also writes inactive package dummies
under Noah-MP/prec_acc_dt0. The approved private CPU build guards only those
allocated packages: guarded SHA `1c1269fb10b7aa135bf42417f1aa7f7a36f7137ea40cd82769316e8ac5bb4918`,
original Gen2 SHA `1d0f5779fff143361e470e37f89c734497ed5a7bb55d0c53b7e3fccc569dd49b`.
The relink control reproduced the original binary, and normal-start outputs
matched 6/6 files through hour one. This separate reference family is explicit
in every twin receipt. Our port uses the existing package-gated carry seeds:
PREC_ACC is absent when disabled, RUC ACRUNOFF is absent under Noah-MP, and
meaningful Noah-MP runoff buckets initialize to zero. No inactive reset arrays
are created. The bug belongs to WRF, not the input container.

## Noah-MP under cycling

`cycling=.true.` is a non-restart start. WRF's `NOAHMP_INIT` takes its
`IF(.NOT.restart)` branch (module_sf_noahmpdrv.F
2008–2190). That branch sets TV = TG = TAH = TSK (273.15 K with snow), EAH = 2000 Pa,
zeroes CANWAT so that CANLIQ = 0, sets CANICE = FWET = SNEQVO = 0 and ALBOLD = 0.65,
and initializes XSAI from LAI. It reads TSK, SNOW, SNOWH, TSLB, SMOIS and LAI from the
input. The port's land initializer therefore never reads the overwritten input fields
under cycling (`NOAHMP_INIT_OVERWRITES` in `io/noahmp_land_init.py`). It also never
applies the land-cycle warm restore there, even though native files carry the land
marker. The land-only `gpuwrf cycle` with `cycling=.false.` keeps its restart-branch
analogue unchanged.

## First root boundary interval

Pristine WRFDA da_update_bc
544–650, 715–855 and 911–992: replace the initial boundary value with the coupled
warm state; preserve the second boundary VALUE when present, otherwise preserve
`old_first + old_tendency*interval`. Set
`tendency=(preserved_endpoint-new_first)/interval`. Only record zero changes.
Encoded WRF boundary clocks are supported.

da_calc_mu_uv
replicates edge dry mass and averages adjacent masses for U/V.
da_couple
uses `field*(C1*dry_mass+C2)/mapfactor` for winds. T/moisture use C1H/C2H;
PH/W use C1F/C2F; T uses THM for USE_THETA_M1. East/north widths are reversed.
WRFDA's default list is U/V/T/PH/QVAPOR/MU; the same Registry coupling explicitly
extends to other carried fields with boundary records. Fresh HT_SHAD/legacy PC
slots are retained; unrecognized boundary fields refuse. Child forcing comes
from its cycled parent.

The endpoint check uses a derived four-roundoff cancellation bound plus
`eta*(interval+2)/2` for IEEE subnormal rounding (eta is the smallest subnormal).
This handles 36 subnormal May QCLOUD boundary cells without changing tendency
bits. It is an arithmetic reconstruction check, not a physics tolerance.

Documented deviations (critic advisories A2, A3, A5):

- **A2.** A boundary record whose start value is unchanged keeps its original
  tendency bytes. WRFDA would recompute `(second-first)/interval`, which is the
  same value up to rounding.
- **A3.** W, moisture and number boundary records are updated beyond WRFDA's
  default U/V/T/PH/QVAPOR/MU list. This is inert on the root, because W uses
  zero-gradient boundaries and `have_bcs_moist=.false.`.
- **A5.** Resuming a cycled run requires `cycling=.true.` in the resume namelist,
  otherwise it refuses. WRF itself would re-clear buckets and reset XTIME on
  restart plus cycling; the port refuses instead of reproducing that.

## Evidence and outstanding gates

| Gate | Current evidence |
|---|---|
| Existing file-seam tests and deletion controls | 57 PASS, two genuine deletion kills; CPU_FILE_SEAM.json |
| Three-nest cold self-identity | 987/987 arrays BYTE, 4.13 s, 598 MiB RSS, no JAX |
| Original WRFDA coupling | 39/39 coupled arrays BYTE on three real input grids |
| Original WRF rebalance | 36/36 arrays BYTE, three grids and both theta modes; steady 8–23 ms/domain; CPU_REBALANCE.json |
| Native May file provenance | 1125 source arrays BYTE, three nests; WS13_native.json, 8.192 s / 757 MiB RSS, no JAX |
| MYNN original initialization | Final V5: all 40 original cases / six branch-field groups PASS compiled CPU AVX; generic CPU FMA discrepancy recorded; GPU pending |
| CPU-WRF same-input twin | Guarded Gen2: 12/12 history files + 9 restart files, three hours / three nests, rc0; input identity and two-key decomposition diff proven; model wall 1213.29 s on 3 physical cores |
| Default OFF program / helper raw HLO | Swiss b977712f and PROD 321aa651 EXACT; default helper f32/f64 raw HLO BYTE-equal |
| Regression / restart controls | 296 CPU tests PASS; both restart formats round-trip BYTE and refuse missing/inconsistent cycling markers before device restoration |
| Actual loader vs original CPU t0 | 34/54 selected field/domain comparisons BYTE; nested maxima U3.82e-6 m/s, P7.33e-4 Pa, PH3.174e-2 m2/s2, QV2.80e-8 kg/kg; source subnormal differences reported; no new tolerances |
| F1 Noah-MP t0 vs CPU-WRF twin (same inputs) | 69/69 field/domain arrays BYTE: TV, TG, TAH, EAH, CANLIQ, CANICE, FWET, SNEQVO, ALBOLD, XSAI, LAI, ISNOW, TSNO, SNICE, SNLIQ, ZSNSO, SH2O, SMOIS, TSLB, SNOW, SNOWH, TAUSS, TSK on d01–d03. Actual loader plus history t0 mapping, `land_t0_probe.py`, CPU, load 441 s. Deletion tests: removing the gate or re-enabling the warm restore fails 2/2 |
| F2 WRFDA update_low_bdy | Fixture rule test plus refusal; removing the update fails 2/2. Regenerated May case changes only SST, water TSK, TMN, VEGFRA and ALBBCK vs the old case (d01 TSK 4.68 K, TMN 0.84 K); wrfbdy byte-identical. CPU-WRF twin on the new inputs running |
| F3 original-WRF witnesses | Critic mutants M1–M5, M7, M8 each fail on value assertions, 0 setup errors. Cloudy and ice/snow rebalance and MYNN witnesses, compiled WRFDA coupling, start_em W gate. Frozen data in `tests/data/w44_warmstart` with provenance |
| Null-resume / metadata change / new forcing | Separate paired GPU arms pending |
| Three-nest 3 h GPU, finite/CWP>0/boundary/D6 | Pending final source and manager night GO |
| Performance/memory and cached readiness | GPU costs unmeasured; host costs above are not forecast speed claims |

The cycling host work occurs during loading. The optional MYNN moments share the
existing five initialization passes and bounded column tiling; GPU launches, memory
and compile cost require measurement. The first booked GPU job is the fixed
original-init component oracle on 2026-10-24 after Nightly-DONE (~22:00Z). The
three-nest launcher then requires that GPU PASS plus E41, uses the unchanged
CPU-twin inputs, and records an existing on-device census, cold setup cost,
per-file hashes, RSS and VRAM. The primary job is capped at 45 minutes
including cold compilation. A separate optional 15-minute cache job checks
warm zero-compiles and exact output/native bytes. Only GA_START_UTC/GA_END_UTC
are supplied: the component writes NIGHT_COMPONENT_RESULT.json and the primary
writes NIGHT_FORECAST_RESULT.json under the lane artifact directory; each next
job refuses a missing or non-PASS predecessor. Its timing is a CENSUS=1 shared-host
qualification proxy. Paired D6 runs once on the retained cold output; warm
GPU-vs-GPU byte identity is regression evidence. File hashes and component tests do not establish
forecast fidelity or release readiness.
