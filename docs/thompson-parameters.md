# Thompson runtime parameters (research candidate)

This opt-in family lets repeated native Thompson (`mp_physics=8`) experiments
change selected coefficients without selecting a different compiled program for
each value. Keep the enabled parameter names, shapes and dtypes fixed.
Omitting the family uses the existing static program.

These coefficients are **gpuwrf extensions**, not WRF namelist options. WRF
4.7.1 defines them as private REAL PARAMETERs or source literals in
`module_mp_thompson.F`. A CPU-WRF comparison at changed values requires
recompiling that module with the same coefficient or literal changed.

| Name | Default | Units | Meaning and pristine source |
|---|---:|---|---|
| `nt_c` | 100000000 | m⁻³ | Fixed droplet number, :88; 100 cm⁻³. Marine examples: 50000000 and 150000000 |
| `tno` | 5 | m⁻³ | Cooper ice-number prefactor, definition :188, use :2625 |
| `ato` | 0.304 | K⁻¹ | Temperature dependence of Cooper ice nucleation, :189/:2625 |
| `av_c` | 31694600 | m⁻¹ s⁻¹ | Cloud-droplet fall-speed coefficient, :163/:3661 |
| `av_i` | 1493.9 | s⁻¹ | Ice mass/number fall-speed coefficient, :161/:3681/:3686 |
| `av_r` | 4854 | s⁻¹ | Rain fall-speed coefficient and its collection/evaporation initialization, :143/:786–788/:801 |
| `autoconversion_threshold` | 0.00001 | kg m⁻³ | Cloud-water **density** guard, source literal :2180; not a mixing-ratio threshold |
| `autoconversion_rate` | 0.027 | dimensionless | Actual autoconversion coefficient, source literal :2186; not a multiplier relative to the default |

Values must be finite positive numeric scalars representable in float32.
The exception is `autoconversion_rate`, which also accepts exactly zero.
`tno`, `ato`, `av_c`, `av_i` and `av_r` are limited to 0.5–2 times their WRF
defaults, inclusive after REAL32 rounding. This initial research envelope
prevents mistyped coefficients from creating excessive sedimentation loops
or ice-nucleation amplification. WRF itself does not declare these bounds.
The threshold must be at least 1e-12 kg m⁻³; smaller values would expose an
undefined cloud-slope branch in the original equations.
`nt_c` admits 2–1999000000 m⁻³, matching the cloud-number floor/cap and avoiding
overflow in WRF's integer shape calculation. The qualified marine interval is
50–150 cm⁻³; multiply cm⁻³ by one million for this input. The pristine default
is 100 cm⁻³; 300 cm⁻³ is WRF's suggested continental value, not its default.
The family requires the native WRF-ordered mp=8 path, explicit sedimentation
and the WRF constant set. Aerosol-aware mp=28, hail-aware mp=38, optional
column tiling and inactive microphysics are refused.

The public runtime binder currently accepts only `nt_c`, `tno`, `ato` and
`av_c`. It explicitly refuses `av_i`, `av_r` and both autoconversion names
until their own GPU, compiled-kernel resource and cost checks pass. Those four experimental controls
remain available only to the low-level qualification tests; their presence
in the table is not a public runtime-support claim.

Use the shared `--runtime-parameters PARAMETERS.json` option:

```json
{
  "schema": "gpuwrf-runtime-parameters-v1",
  "domains": {
    "d01": {
      "thompson_parameters": {"nt_c": 50000000.0, "tno": 5.0, "ato": 0.304, "av_c": 31694600.0}
    }
  }
}
```

Only named domains and coefficients are overridden. A different selected
name set selects another program. Normal checkpoint resume requires the same
parameter values; a declared branch uses the shared branch validation rules.
The four admitted names passed component and short two-domain GPU root
checks and remain experimental. Three-nest D6, station comparisons, 24–72-hour
drift and full release qualification are not covered. The family uses the shared file and checkpoint handling
described under [runtime parameters](runtime-parameters.md). Actual GPU ON/changed values reused the same keys with no new native blobs;
first enabling a family can still require a cold compile.

With the family off, the lowered GPU programs for the bundled Swiss case and the
two-domain Canary Islands case are identical (checked on the CPU). On the qualified
24-column CPU corpus, TNO/ATO at defaults are byte-identical. Enabling default
AV_C changes two cloud-water cells and two cloud-water sedimentation values by
at most two WRF REAL ulps through constant-folding order. Maximum differences
are 1.77636e-15 kg/kg and 3.63798e-12 mm. The latter is a float32 calculation
widened into float64 storage, not a two-float64-ulp claim. Changed coefficients
pass the existing pristine-WRF increment tolerances. These CPU results do not
establish forecast sensitivity, station skill or GPU performance. With `nt_c`
enabled at its default, rain number differs in two cells by at most three
REAL ulps (3.05176e-5 kg⁻¹); other changed fields remain within two REAL ulps.
The cause is localized to constant folding in the cloud moments. Changed
50/150 cm⁻³ runs pass the original WRF microphysics tolerances. Cloud and ice
effective radii are byte-exact to the recompiled WRF references; the inherited
snow-radius error is unchanged and within its existing bound.

The real GPU column probe for `nt_c`, `tno`, `ato` and `av_c` produced
20/20 byte-identical output leaves at defaults. All five changed-value
pristine-WRF checks passed, with zero nonfinite or inadmissible events in the
qualified probes. This is component evidence; the whole-forecast GPU cost and
compiled-kernel resource checks remain open.

The two autoconversion controls and `av_i`/`av_r` have separate CPU checks;
they are not covered by that four-parameter GPU result. Default autoconversion
controls are byte-identical on the 24-column corpus. The isolated ice and rain
coefficients produce localized association differences of up to four and six
WRF REAL ulps, respectively: `av_i` changes qi by up to 3 and Ni by up to 4;
`av_r` changes qr by up to 3 and Nr by up to 6. The ice-precipitation result
has a one-REAL-ulp difference widened into float64 storage. Changed values
pass their original WRF checks.

All eight names enabled together at their WRF defaults (CPU, same 24 columns)
change qc by up to 2, qr 3, qi 3, Ni 4, Nr 6 and graupel number 1 WRF REAL
ulps in a few dozen cells. That is the union of the isolated classes, with no
new interaction. Replacing only the eight traced constants and their derived
REAL prefixes with the static values restores all 20 outputs byte-for-byte,
so the cause is constant folding and association, not changed physics. For
every field the original WRF routine returns (water vapour, the five
hydrometeor mixing ratios, ice and rain number, potential temperature, and the
rain, snow and graupel amounts per step), the largest difference stays within
the largest change pristine WRF itself shows when each input is nudged by one
REAL ulp (25-member ensemble, same columns). Graupel number and the separate
cloud-water and ice precipitation classes are not covered by that comparison.
This is a per-field bound: one cloud-ice-number cell and three
rain-number cells exceed the stricter cell-by-cell envelope of that ensemble. GPU behavior and
cost of `av_i`, `av_r` and the autoconversion controls remain unqualified, so
the public binder refuses them.
`av_r` updates all dependent scalar initialization coefficients as well as
both sedimentation moments. It does not rescale collection tables whose
active rain-velocity formula is independent of `av_r`.

Changing `nt_c` updates the integer gamma-shape choice, all dependent cloud
moments and freezing-table selection, and the held cloud effective radius
used by LW and SW radiation. WRF builds the complete gamma and freezing grids
independently of `nt_c`; an additional 2.39 MB bundled table retains those
original values. Scalar closure work runs before the column kernel rather
than repeating its DOUBLE logarithm in every column.

`ccn_conc` has no Thompson consumer and is refused here. Snow fall-speed control
remains follow-up work.
Snow's `av_s` also changes collision/collection tables; changing only its
sedimentation use would not reproduce a changed WRF `av_s` PARAMETER.

The evolving evidence, integration dependencies and release-note disclosure
are in [THPARAMS.md](../proofs/w44_thparams/THPARAMS.md).

## Compare with original CPU-WRF

Recompile `module_mp_thompson.F` with the same coefficient or source literal
changed, and keep the original build for the control. Hydrometeors, number
concentrations, RAINNC, CLDFRA, SWDOWN and GLW are already in WRF's default
history. The effective radii that radiation uses are restart-only; request
them in `iofields_filename`:

```text
+:h:0:RE_CLOUD,RE_ICE,RE_SNOW
```

Compare radii only where both builds compute them for radiation.


## Short-root GPU cost scope

The final profile has three exact OFF/ON NVTX pairs. Kernels changed from 3409
to 3416 and summed kernel time from 33.510818 to 33.811965 ms (+0.899%); copy rows
stay 620 in each arm. With three profile samples and no A/A floor this is not a
precise slowdown or speedup claim. Arguments add 16 B and temporary storage 32 B
per domain; output/alias sizes do not change. The combined JAX-live peak is not
an individual-process framebuffer peak. Baseline root-replay transfers remain;
this does not prove a transfer-free timestep or global device-operation limit.

The [release facts](../proofs/w44_docs/receipts/V044_RELEASE_FACTS.json) bind the
final four-name gate evidence. Thompson is microphysics, not PBL mixing. The
unqualified ice/rain fall-law and autoconversion extensions remain refused.
