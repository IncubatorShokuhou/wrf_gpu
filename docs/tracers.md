# Passive tracers

**Status in v0.4.0: Validated 24 h.** Passed a 24-hour comparison with the original CPU WRF in the v0.4.0 release check, in one tested single-domain configuration. Other combinations and nested domains are not validated. Its compiled GPU code exceeds the project's code-quality limits for speed and robustness; it runs correctly, but its speed has not been measured. The tracer concentrations themselves have no acceptance limit; WRF-Chem and two-way tracer feedback are not supported.

Passive tracers follow resolved scalar transport and generic WRF PBL diffusion.
They do not change the meteorological state. This feature is opt-in and does not
require WRF-Chem. It is not a chemistry, aerosol microphysics, or pollen biology
model.

In a short three-step CPU comparison with the original CPU WRF, the largest
absolute tracer error was `1.3697e-4`, above that test's fixed `1e-4` limit. The
positive-definite transport operators match the original WRF code exactly at the
worst positions; the remaining difference comes from shared acoustic and
dynamics inputs. Tracer checks therefore rely on operator comparisons and mass
conservation. See the
[qualification evidence](../proofs/u2_tracers/QUALIFICATION.json).

WRF `tracer_opt = 2` supplies eight dimensionless WRF test fields. These use the
non-Chem WRF definitions and have no CHEM sinks. Custom tracers may be used
alongside these built-in fields. Their case-sensitive WRF NetCDF names are
`tr17_1` through `tr17_8` (lowercase), as quoted in WRF's Registry.

Existing input fields are retained; when all eight are absent the real.exe
template initializes the first four in a 9×9 patch at the lowest layer. A partial
group is refused. Modes 1, 3, and 4 require WRF-Chem and are refused. Supported
choices are `tracer_adv_opt=0/1/2`, RK3 with acoustic sumflux, `diff_opt=0` or
`diff_opt=1,km_opt=4`, and `bldt=0`. PBL mixing uses the original WRF `diff4d`
with MYNN exchange coefficients; `tracer_pblmix=0` disables it. Other PBL schemes
require `tracer_pblmix=0`. Root inflow is zero and outflow copies the interior.
Children require the same ordered species list and parent boundary forcing.
The supported nested path is one-way (`feedback=0`). Two-way tracer feedback
is refused because the current feedback registry does not include these fields.

## Custom tracers

Set `gpuwrf_tracer_file` in `&dynamics` to the tracer JSON file:

```fortran
&dynamics
 tracer_opt = 2,
 gpuwrf_tracer_file = 'tracers.json',
/
```

The custom scalar is a mass mixing ratio in **kg tracer / kg dry air**. All
custom tracers begin at zero. Species names must be uppercase and match
`[A-Z][A-Z0-9_]*`; names colliding with WRF fields, the `Q*` moisture namespace,
or `TR17_*` built-in tracer namespace are rejected. Names must be unique. There
is no fixed software limit on the number of custom species; each adds scalar
transport storage and computation.

```json
{
  "version": 1,
  "species": [
    {
      "name": "SMOKE",
      "decay_rate_s": 0.00001,
      "deposition_velocity_m_s": 0.005,
      "wet_scavenging_per_mm": 0.1
    },
    {"name": "POLLEN"}
  ],
  "emissions": [
    {
      "species": "SMOKE",
      "start_s": 0,
      "end_s": 3600,
      "x": 10,
      "y": 10,
      "rate_kg_s": 1
    },
    {
      "species": "POLLEN",
      "start_s": 1800,
      "end_s": 7200,
      "x0": 0,
      "x1": 10,
      "y0": 0,
      "y1": 10,
      "flux_kg_m2_s": 0.000001
    }
  ]
}
```

All loss coefficients default to zero and must be finite and nonnegative.
Coefficients, emission rates, and interval endpoints must also remain finite
when converted to the model's REAL32 format.

| Key | Units | Analytic loss |
| --- | --- | --- |
| `decay_rate_s` | s⁻¹ | Multiply mixing ratio by `exp(-decay_rate_s × dt)` |
| `deposition_velocity_m_s` | m s⁻¹ | Lowest layer only: multiply by `exp(-deposition_velocity_m_s × dt / dz)` |
| `wet_scavenging_per_mm` | mm⁻¹ | Multiply by `exp(-wet_scavenging_per_mm × precipitation_increment_mm)` |

Here `dz` is the geometric depth of the bottom model layer in metres, `dt` is
seconds, and precipitation is the accumulated precipitation increment over the
step, in millimetres. These are user-chosen bulk removal laws, not WRF-Chem
chemical or aerosol sink parameterizations.

Emission time is forecast elapsed seconds. Intervals are **`[start_s, end_s)`**,
with `end_s > start_s >= 0`; a step crossing an endpoint uses only the interval
overlap. Every source injects into the lowest model layer. Point source `x,y`
are zero-based horizontal **mass-grid** indices, and `rate_kg_s` is the total
tracer mass emitted per second into that cell. Area source bounds are
**`[x0,x1) × [y0,y1)`** and `flux_kg_m2_s` is mass per horizontal square metre
per second in each included cell. Cell area and dry-air layer mass convert
these sources to mixing ratio; an area source's total rate depends on covered
cell areas. Overlapping sources add.

The custom source/sink operator runs after resolved transport. Combined losses
use one exponential; each sink receives its proportional share of removed mass.
Emissions are then added, so freshly emitted material starts losing mass in the
following step. No scalar clipping is applied. The resident cumulative budget
records emitted, decayed, dry-deposited and wet-scavenged kilograms per species.
Boundary exchange and WRF's fixed top-level PBL reservoir are separate from
those terms; a closed-domain conservation check must account for the selected
transport and mixing boundaries.

Indices refer to the target domain grid, including its indexed boundary cells;
they are not latitude/longitude, staggered wind indices, or inclusive upper
bounds. Every rectangle must fit its domain. Source geometry is validated
against `(nz,ny,nx)` before stepping. The parser rejects unknown keys, duplicate
JSON keys, unknown species, mixed point/area fields, invalid versions, nonfinite
or negative rates, and empty or out-of-bounds rectangles. It never clamps input
indices or silently ignores misspelled configuration. `emissions` may be
omitted; `species` must be a nonempty list.

Run admission also refuses source products, mapped-cell amounts, concentrations
and cumulative budgets that cannot fit finite REAL32 values. The complete
emission schedule must fit, including material that may later enter a loss budget,
even when the requested forecast ends before the emission interval.

The host parser records the SHA256 of the exact JSON bytes as `source_sha256`
for provenance. Parsing allocates no model arrays and imports no JAX. Enabling
custom species adds per-species transport and loss work; no speed or memory
overhead claim is implied by host-parser checks.

## Output and restart

Ordinary WRF history and auxiliary history subsets can include the active
three-dimensional tracer variables by name. Built-in fields are dimensionless;
custom fields use `kg kg-1`. They share the history frame's time and mass-grid
coordinates. Nonfinite active tracer values refuse output, including species
omitted by a history subset. The separate regional `gpuwrf.streams.json` channels currently
accept surface fields only and do not accept tracer variables.

Full-carry pickle, NetCDF, native restart-store, and runtime checkpoint formats
preserve tracer concentrations, held PBL tendencies, nested boundary records,
and cumulative source/sink budgets. Emission intervals retain the forecast
elapsed clock on continuation. Restart admission binds the species, exact JSON
content, source/sink coefficients, transport settings, and timestep; changing
these requires a new run. State-only checkpoints cannot preserve active tracers
and are refused.
