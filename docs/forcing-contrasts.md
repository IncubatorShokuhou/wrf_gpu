# Separate initialization, continued forcing and SST effects

`gpuwrf optimize contrasts` prepares discrete, paired experiments from private
WRF input directories and runs the original control unchanged. It checks the
files and fields read by the engine before accepting a completed member.
This CPU-tested workflow is experimental; warm-start GPU qualification and
forecast skill remain separate gates.

ALISIOS prepares its WN3 surface plus AIFS three-dimensional atmosphere,
lateral boundaries and SST through its own source adapters/WPS. Supply admitted
facade copies to this workflow. It refuses ALISIOS data roots. It does not
download, remap or manufacture alternative forcing.

Create a study with one issue time, grid, physics and horizon:

```json
{
  "schema": "gpuwrf.optimize.contrasts.v1",
  "hours": 24,
  "member_timeout_s": 1800,
  "control": {"id": "control", "input_dir": "private/control"},
  "arms": [
    {"id": "ic-only", "kind": "ic", "input_dir": "private/alternative-ic"},
    {"id": "lbc-only", "kind": "lbc", "input_dir": "private/alternative-lbc"},
    {"id": "warm", "kind": "warm-start", "input_dir": "private/cycled"},
    {"id": "sst-held", "kind": "sst-held", "input_dir": "private/held-sst"}
  ]
}
```

Paths resolve relative to the study. Each input directory needs its own
`namelist.input`, `wrfinput_dNN`, `wrfbdy_d01` and multi-record
`wrflowinp_dNN`. All domains use `sst_update=1` and the existing NetCDF aux4
reader. The integer horizon must contain complete root timesteps and have
full lateral-boundary coverage. Each child uses live parent boundaries;
the file LBC contrast applies to d01 and propagates through the nests.

- `ic`: change only atmospheric prognostic fields in `wrfinput`. Continued
  LBCs, SST, static/base fields, land and namelist stay the same.
- `lbc`: change only values/tendencies in `wrfbdy_d01`. This separates the
  effect of continued forcing from the separately declared IC experiment.
- `warm-start`: use [atmospheric cycling](atmospheric-warm-start.md) to prepare
  all nests, with `cycling=.true.` and valid field-hashed transfer provenance.
  WRFDA's first boundary record adjustment is allowed; later LBC records and
  aux4 remain unchanged. This is file initialization, not an exact checkpoint
  continuation. Physical differences are limited to the transported atmosphere
  (U/V/W/T/THM/P/PH/MU, water species, QNICE/QNRAIN and QKE), with transport
  provenance and a real loader readback required for each changed field.
  Terrain/base, SST, land and unknown physical changes refuse, even alongside a
  valid atmospheric transfer. A native cycle that changes these fields is
  unavailable for this restricted contrast. WRF cycling reinitialization still
  applies; this is not an exact continuing-state intervention.
  Added U10/V10 history diagnostics, TITLE and extra history dimensions may be
  retained. Added diagnostics are labelled `retained-diagnostic` in the plan
  and `RETAINED_ONLY_NOT_CONSUMER_PROOF` in the receipt; they do not establish
  consumption of a changed physical input. Changes to pre-existing diagnostic
  values or arbitrary other added fields refuse.
- `sst-held`: repeat the control's first SST grid at every aux4 time. Preserve
  its times, SEAICE, VEGFRA and ALBBCK and every other input exactly.
  Setting `sst_update=0` would also suppress the other aux4 fields and is
  refused here. Preparing the held-series file is the caller's responsibility.

No-change arms, foreign input metadata, changed physics and mixed undeclared
changes are refused. The control is the original cold case; plan and run never
rewrite it. Input byte hashes are checked again before and after every member.

```bash
JAX_PLATFORMS=cpu gpuwrf optimize contrasts plan --study contrasts.json --output plan.json
gpuwrf optimize contrasts run --study contrasts.json --output runs/contrasts
```

The second command executes real forecasts. On the shared development machine,
run it through `scripts/with_gpu_lock.sh` with the manager's lease/window.
The native child inherits only the validated official non-stdio lock descriptor
via `pass_fds`; its existing holder/token/descriptor preflight stays in force.
CPU execution without a lock forwards no descriptors and does not inspect the
official lock file.
Members run sequentially in fresh processes, with a hard timeout of at most
1,800 seconds each. An external optimizer wave/registry controller must book
the run and register it; this command neither grants an external GO nor sets
ALISIOS admission/skill gates. It does not retry or resume a partial wave.
Keep completed receipts and choose a new output directory after a failure.
`planned_case_hours` includes the unchanged control. It produces unscored
paired arms, for the wave controller's common-support impact matrix and
non-degradation tests; it does not choose a winner or mix these arms into EKI.

## Consumer receipts

`plan.json` records each file SHA256, field schema and record-value SHA256,
plus exactly which fields differ from the control. `ARM/consumer.json`
records the input/namelist identities, actual platform, elapsed member time,
and field/record readback hashes returned through the original engine's
IC materializer, boundary decoder and aux4 loader. Numeric hashes use
little-endian float64 so lossless float32-to-float64 decoding compares exactly.
Masked/nonfinite consumed values refuse. Original dtype and dimensions remain
in the plan. No field values or process environment are printed.

Observation runs only in the isolated contrast member. It reads the host arrays
already returned by the loaders before device placement. It adds no timestep
hooks, carry leaves, kernels or device-to-host transfers. Normal forecasts do
not install the observer. A zero process exit without every required read is
refused, as are missing/foreign receipts, wrong readbacks or changed inputs.
`result.json` is published only after every member has a validated receipt.

These prove **loader ingress**, including which full SST series entered the
engine; they do not assert a read at every timestep or prove each later
application. WRF reads aux4 before solve and applies SST after radiation in
the surface phase (D15/E37); that existing implementation is unchanged.
They are provenance evidence, not a WRF oracle, CPU-WRF fidelity or improved
forecast-skill result. Stub tests identify their platform as a stub explicitly.

Planning and validation stream one host variable/record at a time; they hash
each input during admission and again at execution seams. This adds CPU and
disk-read cost proportional to input size, without input copies or a retained
array inventory. Member compile/readiness, VRAM and forecast wall time have
not been measured by this CPU lane.
