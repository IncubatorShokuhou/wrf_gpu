# Checked initialization overrides

`gpuwrf override` writes new copies of per-nest `wrfinput_dNN` and optional
`wrflowinp_dNN` files. Originals are opened read-only. All edits and checks run
on the host; the forecast program and timestep loop do not change.

```sh
gpuwrf override --input-dir original-inputs --output-dir edited-inputs --spec overrides.yaml
```

The output directory must not exist. It contains the copied input files and
`OVERRIDE_PROTOCOL.json`. It is published atomically only after validation.
The tool does not copy `namelist.input`, `wrfbdy_d01`, tables or other case files;
retain those from the original case when assembling a runnable forecast.
An empty spec (`{}`) gives byte-for-byte identical input file copies.

## Spec

```yaml
overrides:
  - nest: 2
    field: SST
    op: add
    values: 1.5
    mask: water
  - nest: 2
    field: SMOIS
    op: scale
    values: 0.95
    mask: land
  - nest: 3
    field: TSK
    op: set
    file: new_tsk.npy
    mask: {file: selected_cells.npz, variable: selected}
```

`nest` is an integer (1 means d01); `op` is `set` (default), `scale` or `add`.
Operations run in spec order. `target: wrflowinp` selects the forcing file
instead of the default `wrfinput`. Supported aux4 edits are SST, VEGFRA and
ALBBCK, across every supplied forcing record.

Supply exactly one of `values` (a scalar or an exact-shape nested array) or
`file` (NPY, NPZ or NetCDF). A singleton Time dimension may be omitted from
values. `variable` selects an NPZ/NetCDF variable; it defaults to the edited
field name. File paths resolve relative to the spec, never the input directory.
Pickled NumPy objects, packed NetCDF targets, missing fields, integer truncation,
NaN/Inf and float-storage overflow are refused.

`mask` is optional: `land`, `water`, a boolean/0/1 array or
`{file: path, variable: name}`. File masks default to variable `mask`.
Horizontal masks broadcast over levels and Time; exact field masks also work.
Named masks use the matching wrfinput LANDMASK, including for aux4. Values and
masks are checked before casting. Missing/unknown spec keys refuse.

## Fields and checks

Atmospheric fields: T (dry potential-temperature perturbation in K), P
(perturbation pressure in Pa), QVAPOR, QCLOUD, QRAIN, QICE, QSNOW, QGRAUP,
QHAIL and supported hydrometeor number/volume fields. Surface/land fields:
SST, TSK, SMOIS, TSLB, LAI, VEGFRA, LU_INDEX/IVGTYP, ISLTYP, EMISS, ALBBCK,
ZNT. The field must already exist; units and storage schema are preserved.
**VEGFRA uses percent (0–100), not a fraction.**

Checks cover nonnegative moisture, RH over liquid water, positive total dry
mass and geopotential layer thickness, soil porosity, MODIS classes,
land/water masks and basic physical ranges. RH uses WRF's dry T convention,
P+PB, R_D/CP=2/7 and Bolton saturation constants. Its default bound is 1.0
and the chosen `max_relative_humidity` is recorded in the spec/protocol.
Soil porosity comes from the STAS block of the actual `SOILPARM.TBL`, whose
path/hash is recorded. The pristine Noah-MP non-seaice glacier SMOIS=1
initialization is allowed; other land columns obey their per-ISLTYP cap.
LU_INDEX and IVGTYP must agree if both exist; edit both together. MODIS
water/lake categories 17/21 must agree with LANDMASK and XLAND.

T/P/QV and other moist-mass edits are refused because the tool does not yet
implement real.exe-consistent PH/MU rebalance. An intentional unbalanced
experiment can explicitly waive this check:

```yaml
overrides:
  - {nest: 1, field: T, op: add, values: 0.1}
  - {nest: 1, field: QVAPOR, op: scale, values: 0.99}
waivers:
  hydrostatic_consistency: "Controlled initialization sensitivity; balance is not asserted"
```

Other waivable checks: `moisture_positivity`, `supersaturation_bound`,
`soil_porosity`, `land_water_mask`, `modis_classes`, `field_ranges`.
Each waiver requires a nonempty reason. A waived failure is recorded as
`WAIVED`, never `PASS`. Schema, nonfinite/fill data, positive mass/geometry,
source transport identity and HGT/base-state editing refusals cannot be waived.
No clipping, saturation adjustment or silent repair is performed.

Under Noah-MP (`--surface-physics 4`, the default), EMISS/ALBBCK/ZNT are
recomputed. These edits refuse with **NO EFFECT under Noah-MP; use
--surface-controls**. `--force-no-effect` permits the storage edit with a warning
and a recorded protocol flag; it does not make the edit affect Noah-MP.
LAI and VEGFRA effects also depend on the configured Noah-MP DVEG option:
wrfinput LAI is used for 7/8/9, VEGFRA for 1/6/7; DVEG=4 uses SHDMAX.

## Protocol and warm-start API

For every copied file the protocol contains original/output SHA-256 and paths,
and each edit has before/after storage-array SHA-256, dtype/shape, changed-cell
count, values/mask provenance (including external file hashes), check outcomes,
waivers and no-effect warnings. Refusal leaves no published output directory.

The shared, host-only API is:

```python
from gpuwrf.io.wrfinput_consistency import validate_wrfinput, WrfinputConsistencyError
report = validate_wrfinput(path, changed_fields={"SMOIS": changed_boolean_array})
```

`changed_fields` can instead be a sequence of names (all cells are then treated
as changed). The report has `passed`, `checks` and `waivers`;
`WrfinputConsistencyError.report` contains the failed checks. `soil_table` can
select an explicit STAS table. `surface_physics` defaults to 4.

For in-memory checkpoint branching, use the same checks directly:

```python
from gpuwrf.io.wrfinput_consistency import validate_fields
report = validate_fields(
    fields,  # WRF field names -> host NumPy arrays, including Time=1
    {"dimensions": dimension_lengths,
     "variable_dimensions": variable_dimension_names,
     "attributes": global_attributes},
    changed_fields=changed_cell_masks, waivers=waiver_reasons,
    surface_physics=4, checkpoint=True,
)
```

`variable_dimensions` maps each field to its WRF dimension-name tuple;
`dimensions` maps names to integer lengths. Optional `variable_attributes`
maps fields to their attributes, including `_FillValue`. No temporary NetCDF
adapter is needed; the file wrapper reads fields lazily and array checks reuse
the existing payload. Neither path mutates inputs or metadata.

Controlled Noah-MP porosity can be checked directly with the shared predicate:

```python
from gpuwrf.io.wrfinput_consistency import validate_soil_porosity
ok = validate_soil_porosity(
    smois, effective_smcmax, landmask, lu_index,
    seaice=seaice, surface_physics=4, sh2o=sh2o,
)
```

It returns a Python boolean. Caps may be scalar, cell or layer arrays that
broadcast to SMOIS. For a one-dimensional WRF class table, supply
`isltyp=soil_classes`; indexing is one-based with an unused slot 0. Supply the
effective cap, including urban overrides, in the Noah-MP consumer's precision.
The helper reads no tables, changes no arrays and introduces no tolerance.
Invalid/nonfinite/masked inputs or incompatible shapes return `False`.
Land requires `0 <= SMOIS <= cap` and, when supplied,
`0 <= SH2O <= SMOIS` and `SH2O <= cap`. The existing Noah-MP glacier15,
SEAICE<=0 cap=1 rule is preserved. The table-backed validator delegates to this
same helper; its public signatures are unchanged.

Checkpoint mode additionally admits U/V/W/PH/QKE and Noah-MP SH2O, TSNO,
SNICE/SNLIQ, SNOWH/SNOW, CANLIQ/CANICE, TV/TG/TAH and SAI. MU and base/static
edits remain refused. A PH-only edit must keep valid geometry and carry the
explicit `hydrostatic_consistency` waiver. Liquid soil water must remain between
zero and SMOIS; edited snow/canopy stores and QKE must be nonnegative, and active
temperatures must be positive. The actual edited footprint must be on land.
An unchanged checkpoint receives finite/schema checks without re-admitting its
existing supersaturation as an initialization analysis.

For the atmospheric warm-start consumer, `transport_reference=path` and
`transport_frame=index` verify retained source T, winds and moisture bytes,
P+PB/PH+PHB/MU+MUB totals, and roundoff-only HGT/PB/PHB/MUB changes. Current
restart slots U/V/W/PH/MU/THM_2 are supported and ambiguous aliases refuse.
This representation transport preserves a model state's nonhydrostatic
residual; it does not assert hydrostatic equilibrium. After source storage and
total checks succeed, inherited negative moisture/number values and liquid RH
exceedances are reported without changing them or introducing waivers. Finite
storage, schema and positive mass/layer geometry remain required.

The receipt adds `inherited_source.moisture[field]` with min/max, cell count,
negative count and horizontal outer-ring negative count. Its
`relative_humidity_over_liquid` entry records min/max, supersaturated count,
outer-ring count, count above the selected bound and undefined diagnostic count.
Diagnostics use the verified transported state; pressure totals may differ from
the reference within the documented storage roundoff. Where hybrid coefficients
exist, the start_em inverse-density/EOS residual remains diagnostic only.

Source transport must verify the complete source field/storage and total
contract. A changed field cannot gain this admission by merely supplying
`transport_reference` or an empty changed-cell mask. Source mismatch refuses;
strict positivity, RH and balance checks still run for edits. Ordinary override
and checkpoint branch admission retain their strict checks and explicit-waiver
protocol. The array API accepts `transport_reference=(source_fields, source_meta)`.

## Qualification status

CPU known-answer, source transport, transactional refusal, whole-file no-op,
NaN/Inf and backend-dark CLI tests: 88 passed on `3701b5e2c9` (OV04).
Existing CLI/default/offline consumer tests: 30 passed (OV05). Removing each
of 11 mandatory checks causes genuine assertion failures, with zero setup
errors (OV07). Unit checks use synthetic real-structured files and pristine
STAS parameters; they do not establish forecast fidelity.

Real original-CPU-WRF 0227 three-nest inputs also pass (OV08/OV10): empty
specs preserve all three files byte-for-byte; QV×0.99 and T+0.05 K, with an
explicit hydrostatic waiver, change exactly the requested storage values.
All 329 variables and 92 global attributes per domain retain their schema and
metadata; every other field retains its storage bytes. The sealed-facade host
preflight measured 0.421 s for the empty operation and 0.914 s for QV/T, with
541 MiB process peak RSS including guard/import setup. These are host file
processing measurements, not forecast speed measurements.

All 929 existing numerical/IO source files are byte-identical to the lane's
base. The other two existing files change only offline command registration
(OV09). Prior default-program qualification is carried through that source
identity proof; no fresh GPU identity/speed result is claimed.

The ALISIOS S3 list and actual CPU-WRF run remain manager follow-ups. The
0227 three-nest 3 h QV/T GPU smoke is prepared and CPU dry-tested, and waits
for the manager's explicit GO (WAIT at 2026-10-23T18:12:33Z). Full lane/release
qualification is pending that smoke; host/file review is accepted as below.

The shared array API shipped in `8bacb7e0c5` (114 tests passed). CR03 corrections
in `51a672a8be` recheck retained SMOIS after LU_INDEX/IVGTYP edits and explain
ALBEDO/SH2O recomputation in refusals. The final delta passes 153 producer/critic
checks, three NetCDF locking consumers and two exact-guard deletion controls.
The same independent critic approved **MERGE, host/file scope only** on
2026-10-23; their 59 array/differential checks also passed. The default model
identity evidence is reused because numerical source remains unchanged.

Changed-checker measurements on the three real 0227 domains: file validation
0.556 s total; existing in-memory payload validation 0.072 s total, excluding
payload loading; peak process RSS 490 MiB. These measure host admission only.
The backend-dark CLI probe also passes with parent PYTHONPATH unset after the
test-only portability correction `c721799d88`.

The GPU launcher is prepared for the manager's nightly sequencer, with no
interactive inputs, lock timeout 1 s and a 20 min cap. It runs the frozen
three-nest QV/T smoke, E41 and finite-output checks, then cleans successful
disposable caches/outputs. No GPU result is claimed before its receipts exist.

Manager-authorized source admission in `06497b56cc` passes 134 tests, including
unchanged negative QNRAIN and RH 1.0049 with exact storage preservation, strict
negative/RH/balance edit refusals, and unchanged finite/schema/geometry gates.
Deleting positivity enforcement or source-proof verification causes five and
three genuine assertion failures respectively, with zero setup errors. The
host source-proof/extrema checker took 0.500 s across three real-case-shaped
paired input copies, excluding payload loading; process peak RSS was 582 MiB,
including the paired payloads/imports. This measures checker cost and does not
claim native May source admission or forecast validation; the warm-start lane
owns that real-source consumer gate. The same independent critic accepted the
source-admission delta; the manager merged it as main `c15f937487`, with 155
post-merge tests passing.

The explicit-cap helper (`d49b259636`) passes 177 tests, including all existing
admission checks and 43 helper/delegation cases. Removing porosity or liquid
bounds causes 10 and 4 genuine assertion failures, with zero setup errors.
Three staged soil grids with an example controlled cap take 0.927 ms total for
the host check, excluding loading; process peak RSS is 196 MiB including
imports/input arrays. This is a helper-cost measurement, not a control-fidelity
or forecast-speed result.
