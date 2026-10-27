# Carry a previous atmosphere into a new forecast

**Experimental, new in v0.4.4.** Component checks against the original WRF and
WRFDA code pass on the CPU. The CPU-reference results below identify which
generation of prepared inputs was tested. GPU initialization and paired GPU
forecast checks are still pending.

`gpuwrf cycle` remains land-only unless atmospheric transfer is requested.
Native atmosphere transfer is a gpuwrf file-preparation extension. The
`cycling=.true.` initialization that follows uses WRF's own lifecycle rules.
It prepares a new issue from a previous forecast at the same valid time.

## Workflow

Prepare a fresh real.exe case with the new boundary and lower forcing. Supply
one previous history/restart frame per nest at that exact initialization time:

```bash
JAX_PLATFORMS=cpu gpuwrf cycle --atmosphere --native-atmosphere \
  --source previous/wrfout_d01_VALID_TIME \
  --wrfinput fresh/wrfinput_d01 --output cycled/wrfinput_d01 \
  --wrfbdy fresh/wrfbdy_d01 --boundary-output cycled/wrfbdy_d01 \
  --nest previous/wrfout_d02_VALID_TIME fresh/wrfinput_d02 cycled/wrfinput_d02 \
  --nest previous/wrfout_d03_VALID_TIME fresh/wrfinput_d03 cycled/wrfinput_d03
```

Set `cycling=.true.` in the new namelist's `&time_control` group. Keep its new
`wrflowinp_dNN`, other forcing and case files alongside the emitted inputs.
The source/fresh pair is required; the shipped Swiss cold input alone is not
a complete cycling example. Existing outputs are never overwritten. Consume
only a successful invocation whose JSON record lists every output: interruption
during multi-file publication may leave a partial set.

Native mode carries the source's resolved terrain and base fields
HGT/PB/PHB/MUB with its atmosphere. WRF's `SAVE_TOPO_FROM_REAL=0` history/restart
convention prevents a second nest-terrain blend. Raw real.exe inputs use 1.
The separate non-native overlay mode (`--atmosphere` without `--native-atmosphere`) is restricted to d01 and
roundoff-sized base/terrain differences; material differences refuse.

## What is transferred and initialized

- U/V/W, dry/moist theta representations, P/PH/MU and active moisture/number
  species use their source storage and native staggering. Required missing
  species refuse. Land/snow uses the existing land-cycle admission and masks;
  soil, canopy water and land snow stay cycled, as in WRFDA
  (`update_lsm=.false.`, `keep_snow_wrf=.true.`).
- The lower-boundary fields SST, TMN, VEGFRA, ALBBCK and SEAICE come from the
  fresh real.exe input, as WRFDA's `update_low_bdy` does. Skin temperature stays
  cycled over land and comes from the fresh input over water; snow over water is
  zeroed.
- QKE and the recorded MYNN state are copied exactly. When a history file has
  no advected QKE (`qke_adv`), it is initialized from the source QKE and this
  is recorded in the output's provenance.
- Grid/vertical coordinates, map factors, categories, schemes, nesting and
  source valid time must agree. Fresh issue/step metadata is recorded. Runtime
  elapsed time starts at zero. Input-only fields taken from fresh real.exe,
  reconstructed reference profiles and other exceptions are listed explicitly.
- Unchanged transported state is checked against its source. Inherited tiny
  negative species or supersaturation are reported with counts; the tool does
  not repair them. Actual edits retain strict checks. Finite data, schema,
  positive dry mass and layer geometry remain required.
- The first root boundary value is replaced by the coupled warm state. The
  next boundary endpoint is preserved and the first tendency is recomputed
  using WRFDA's update rule. Later boundary records retain the fresh forcing;
  child forcing comes from the cycled parent.

Runtime cycling currently requires Thompson option 8, MYNN option 5, DFI off,
`HYPSOMETRIC_OPT=2` and `USE_THETA_M=0/1`. Unmarked cycling inputs, conflicting
restart aliases and unsupported configurations refuse. Restart markers
distinguish cycled initialization from an ordinary cold start.

WRF cycling rebalances pressure/geopotential and applies its W initialization
rule. With default `use_input_w`, W is rebuilt even when the source W is
nonzero. MYNN has a particularly visible initialization rule: it chooses cold
QKE initialization when the incoming column maximum is below 0.0002, then
clears turbulence/cloud fields for non-restart cycling. Columns with higher
incoming QKE therefore begin with zero QKE. Carrying QKE in a file does not
bypass this original-WRF behavior; no keep-QKE extension is included.

Noah-MP takes WRF's non-restart initialization under cycling: canopy and
ground temperatures start from the skin temperature, and canopy water, the wet
fraction and the previous-step snow albedo are reset, exactly as `NOAHMP_INIT` does. The land-only `gpuwrf cycle`
keeps its existing behavior.

Three small documented deviations from WRF remain: a boundary record whose
first value is unchanged keeps its original tendency (WRFDA would recompute the
same value up to rounding); W, moisture and number boundary records are also
updated, which has no effect on the root domain; and resuming a cycled run
requires `cycling=.true.` in the resume namelist, otherwise the run is refused.

This is warm initialization. Integrator scratch and unavailable state are
reinitialized. Use [checkpoint continuation](checkpoint-branching.md) when
the experiment needs the model's continuing held physics and scratch.

## Evidence, comparison and cost

The pinned report records 39/39 original-WRFDA coupled arrays and 36/36 WRF
rebalance arrays byte-identical on the tested grids. Native three-nest file
preparation retains 1,125 source arrays exactly, and a three-nest self-copy
keeps 987/987 arrays byte-identical. Forty original MYNN initialization cases
pass the compiled CPU configuration described in the report; its generic-CPU
floating-point discrepancy remains disclosed. With inputs prepared before the
WRFDA lower-boundary update was added, the model's input loader matches
original CPU-WRF at the first time in 34 of 54 selected field/domain pairs
byte for byte; the largest nested differences are
3.8e-6 m/s in U, 7.3e-4 Pa in P, 3.2e-2 m²/s² in PH and 2.8e-8 kg/kg in water
vapor. The completed three-hour CPU-WRF cycling reference on three nests also
used those earlier inputs and wrote all 12 history and 9 restart files. Those
results do not validate the current tool's regenerated inputs: the
lower-boundary update changes SST, water TSK, TMN, VEGFRA and ALBBCK.

For the regenerated inputs, the Noah-MP land state at the first time matches
the new CPU-WRF reference byte for byte in all 69 field/domain arrays. The
pinned report records that first-time comparison but still lists the new
three-hour reference as running; it does not provide its completion receipt.
With cycling off, the recorded raw-HLO comparison found the default programs
unchanged. These are component and CPU-reference results, not a completed GPU
forecast comparison.

The CPU-WRF references use an explicitly identified private build with guarded
inactive-package bucket resets. Original WRF 4.7.1 writes inactive package
dummies when cycling Noah-MP; the reference build and normal-start byte checks are
documented in the [reference-build report](../proofs/w44_wrfdebug/WRFDEBUG.md).
Do not treat its output as coming from an unmodified binary.

Measured host file preparation is 8.192 s with 757 MiB process RSS on the
recorded three-nest pair; the rebalance component takes 8–23 ms per domain.
The completed CPU-WRF reference with the earlier inputs took 1,213 s of model
time on three physical cores for its three hours; that is that reference run's
cost, not a gpuwrf estimate.
These exclude a GPU forecast. GPU initialization/compile time, extra device
work, peak VRAM and cached readiness remain unmeasured in this report.

For original CPU-WRF comparison, retain native restart data and the exact
forcing. The prognostic atmosphere (U, V, W, T, P, PB, PH, PHB, MU, MUB, the
moisture species and QKE) is already in WRF's default history, so no
`iofields_filename` line is needed for it. Check that scheme-specific number
fields and the [land-cycle fields](cycling.md) are present in both outputs.
History alone does not preserve every restart quantity. Null continuation,
metadata-only changes and new forcing require separate comparisons.

Evidence: the [warm-start report](../proofs/w44_warmstart/WARMSTART.md).
