# Spectral nudging (`grid_fdda=2`)

Spectral FDDA nudges the large horizontal scales of wind, dry potential temperature,
perturbation geopotential and optionally water vapor toward `wrffdda_d0N` analyses.
The WRF REAL32 operator is CPU-oracle-qualified against the unmodified WRF routine
and FFTPACK5. GPU and coupled-forecast qualification remain pending.

| Selection | Implemented scope | Qualification |
|---|---|---|
| `grid_fdda=2` | U/V/T/PH/Q, per-domain analyses, cadence, vertical weights, end ramps, restart | CPU pristine operator/caller oracle; GPU and coupled forecast pending |

```fortran
&fdda
 grid_fdda = 2,
 gfdda_inname = 'wrffdda_d<domain>',
 gfdda_interval_m = 60,
 gfdda_end_h = 6,
 io_form_gfdda = 2,
 guv = 0.0003,
 gt = 0.0003,
 gph = 0.0003,
 gq = 0.0,
 xwavenum = 2,
 ywavenum = 2,
 fgdt = 0,
/
```

Coefficients are inverse seconds. Each analysis file contains time-bracketing OLD
and NEW U/V/T/Q/PH fields with WRF staggered dimensions; T is dry potential
temperature minus 300 K and PH is perturbation geopotential in m²/s². Prepare the
files with WRF real.exe or an equivalent verified analysis producer. The reader
also accepts WRF's packed `fdda3d` mass dimensions for U/V/PH, padding their
unused outer slots. When a custom producer's files will also be read by CPU-WRF,
include the WRF version/global metadata and per-variable `FieldType=104`;
otherwise CPU-WRF can report SUCCESS while leaving analyses at zero.
The reader
checks interval coverage, shape, missing values and finiteness independently per
domain. Analyses stay resident during stepping. A restart retains the analysis
cursor, both current records and held tendencies, including all three PH fields;
missing spectral carry fields cause a refusal.

Filtering follows WRF's extra zeroed east/north points and FFTPACK mode convention.
`xwavenum=0` or `ywavenum=0` retains the mean along that direction; a negative cutoff
removes all modes. The forward/inverse pair preserves amplitude. Horizontal modes
above either cutoff are removed from the analysis-minus-model difference before
vertical weights and coefficients are applied.

`if_no_pbl_nudging_uv/t/ph/q=1` selects PBL weights and takes precedence over
`if_zfac_uv/t/ph/q=1`. `k_zfac_*` uses WRF's one-based vertical indices;
`dk_zfac_*` sets a linear transition width (default 1, must be positive when used).
PBL restrictions currently require MYNN's held PBLH. `ktrop` halves theta/Q
nudging at the selected level and stops it above; zero disables the lid.
`if_ramping=1` and `dtramp_min` control the signed end ramp. A positive ramp extends
past `gfdda_end_h` and extrapolates the bracketing analysis, as WRF does.

`fgdt` is minutes between tendency updates; zero recomputes every step.
`fgdtzero=1` clears held tendencies between calls according to the WRF driver.
The port preserves pristine WRF's unusual source lifetimes: PH is not mass-coupled
on entry but is decoupled at step exit; Q is mass-coupled on entry and stays coupled
on spectral step exit. Nonzero Q with held tendencies therefore requires particular
care and remains part of pending coupled-forecast qualification. The original WRF
source, rather than a different GPU version, defines this behavior.

Default forecasts (`grid_fdda=0`) and PSU grid FDDA (`grid_fdda=1`) keep their existing
paths. Reference fixtures, oracle build and reproducible probes are in
`proofs/s1_specnudge/`; REAL64 fixtures are precision-promoted diagnostics, while
REAL32 fixtures use original WRF/FFTPACK precisions.

The short Swiss CPU smoke executed three 18 s steps with zero nonfinite values
and zero guard events. Against the original CPU-WRF spectral run, RMSE was
0.000969 K for dry potential temperature, 0.008884/0.005626 m/s for U/V,
0.006986 m/s for W and 6.19e-7 kg/kg for vapor, all within the existing core-field
limits. This is a short legacy CPU smoke, not the 24 h/all-variable release gate.

On the full Swiss 44×42×42 geometry, the CPU component's past-end median fell
from 8.05 ms to 0.071 ms by preserving WRF's early return before FFTs. Active
medians were 8.51/8.23 ms, which do not establish a gain beyond shared-core noise.
Compiler temporary/output estimates stayed at 2.31/1.57 MB; the REAL32 component
had no nonscalar f64 results. These are CPU results, not GPU timing or peak VRAM.

Resident REAL32 records cost eight times the total U/V/T/PH/Q element count in
bytes per OLD/NEW pair, plus twelve times that count for the held carry. For
the two-domain PROD geometry, 24 record pairs occupy about 1.571 GiB plus
0.098 GiB of carry; PH adds about 0.339 GiB relative to PSU FDDA with the same
records. These are logical array totals, not measured peak/alias/workspace use.
`grid_fdda=0` allocates no spectral arrays and retains the released program hash.
