# Offline synthetic radar

`gpuwrf post --product radar` produces an instantaneous 3-D reflectivity,
column composite, echo top and radar VIL. It reads CPU-WRF or gpuwrf history;
the forecast program is unchanged. Multiple samples describe those history
times, not the maximum between them.

```sh
gpuwrf post --input wrfout_d01_* --output radar.nc --product radar
pip install 'gpuwrf[radar]'  # optional full Mie scattering for C/X
gpuwrf post --input wrfout_d01_* --output radar_x.nc --product radar \
  --option 'radar_band="X"' --option 'radar_method="thompson"'
```

The second command requires the Thompson hydrometeors and numbers described
below. Ordinary mp=8 wrfout does not contain the necessary graupel number for
active graupel; the command then reports the missing input. A C/X product is
never obtained by rescaling the S-band dBZ.

| Option | Default | Definition |
|---|---|---|
| `radar_band` | `"S"` | S=0.10 m, C=0.056 m, X=0.032 m wavelength |
| `radar_method` | `"auto"` | Saved S `REFL_10CM` when available; otherwise Thompson for `MP_PHYSICS=8`; other schemes require explicit `"rip"` |
| `radar_echo_dbz` | `18` | Finite echo threshold above the method's reflectivity floor |
| `radar_variable_intercept` | `false` | RIP-only Thompson (2004) variable exponential intercepts |
| `radar_liquid_skin` | `false` | RIP-only liquid dielectric for above-freezing ice |

## S-band methods and qualification

**Saved:** `radar_method="saved"` copies `REFL_10CM` without changing values or
dtype. Its scheme and sampling are those of the producing model. A real
CPU-WRF Swiss three-hour run with `do_radar_ref=1` supplies the parity reference.
This also works for scheme-specific reflectivity from other microphysics.

**Thompson:** `radar_method="thompson"` evaluates the mp=8 distribution and
threshold rules of pristine WRF `phys/module_mp_thompson.F:calc_refl10cm`.
Temperature and pressure are reconstructed from `T+300`, `P+PB`, with the WRF
dry-potential-temperature exponent 287/1004.5. Inputs `QVAPOR`, `QRAIN`,
`QSNOW`, `QGRAUP` are kg/kg; `QNRAIN`, `QNGRAUPEL` are number/kg. Active rain
needs positive `QNRAIN`; graupel above 1e-6 kg/kg needs positive `QNGRAUPEL`.
Missing `QNGRAUPEL` is allowed only when every graupel cell is radar-inactive.
The operator ignores cloud water and cloud ice as pristine calc_refl10cm does.
Hail-aware Thompson mp=38 and active hail are refused.

Rain/graupel have exponential size distributions with number and mass moments;
graupel density is 400 kg/m³. Snow follows the two-gamma Field et al. (2005)
distribution, mass=.069 D², and WRF temperature-dependent moments. The highest
warm raining layer with ice immediately above starts melting. Snow melt
fraction follows WRF's column-mass ratio, bounded 0.05–0.99, with its original
50-bin wet-snow integration. WRF's wet-graupel enhancement is commented out and
is also absent here. The WRF writer floor is −35 dBZ.

The NumPy operator was compared against a separately compiled, verbatim
pristine `calc_refl10cm` plus the complete `module_mp_radar.F`, retaining REAL
rounding and the used initialization constants. Forty 44-level columns cover
24 real Swiss history-field columns and 16 constructed active rain, snow,
graupel, melting and clear regimes. Maximum difference was 3.82e-6 dBZ;
the checked-in fixture pins a 5e-6 dBZ tolerance. Source hashes and compiler
flags are in `proofs/u2_radar/oracle_manifest.json`; the generator stages
copies in a caller-selected build directory.

**Sampling limitation:** mp=8's `ng1d` evolves privately during microphysics and
is not saved in ordinary WRF history. `REFL_10CM` is evaluated at microphysics
exit, before the completed timestep's history state. Thus reconstructed
history reflectivity cannot be claimed byte-identical to the online field,
even when an external number field is supplied. Use saved reflectivity for
exact history parity. The compiled-column oracle qualifies the mathematical
operator; its real-history inputs are not call-site savepoints.

**RIP fallback:** explicitly selecting `radar_method="rip"` computes the NCAR
[wrf-python CALCDBZ](https://github.com/NCAR/wrf-python/blob/v1.3.4.1/fortran/wrf_user_dbz.f90)
operator for other schemes, at S-band only. It uses spherical constant-density
rain/snow/graupel (1000/100/400 kg/m³) with fixed exponential intercepts
8e6/2e7/4e6 m⁻⁴ by default. Its floor is −30 dBZ. If snow is absent, below-freezing
rain becomes snow as CALCDBZ specifies. Sixty combinations are checked against
the original compiled NCAR Fortran, including variable intercepts, liquid skin,
zero/trace/dense species and missing snow. This is a documented generic
diagnostic, not scheme-specific `REFL_10CM` parity.

## C/X-band scattering and assumptions

Thompson C/X uses [miepython](https://github.com/scottprahl/miepython)'s full
Wiscombe/Bohren–Huffman Mie series, not Rayleigh with a wavelength correction.
Radar cross-section is `Qback*pi*D²/4`; equivalent reflectivity is
`lambda⁴/(pi⁵*|Kw|²) * integral(N(D)*sigma_back(D) dD)`, converted from m³ to
mm⁶/m³ with 1e18. `Kw=(mw²−1)/(mw²+2)`.

Rain is a homogeneous water sphere. Dry snow and graupel are homogeneous
effective ice/air spheres with WRF's nested Maxwell-Garnett spheroidal-inclusion
mixing convention and their mass–diameter relations. Wet snow uses WRF's
air/ice/water volume and melt geometry, then full Mie scattering of the
resulting *effective homogeneous sphere*. Water/ice refractive indices follow
WRF's Ray (1972)/Maetzler (1998) formulas at **fixed 0°C**, including WRF literal
rounding. This is a reproducible sphere-model assumption, not measured aggregate
or layered-particle scattering. Large nonspherical drops, aggregate geometry,
water shells and shedding can differ substantially, especially in melting.
Polarimetric fields, attenuation, beam elevation/width, terrain blockage and
multiple scattering are not modeled. Products are intrinsic reflectivity.

Dry PSD integrals use 256 logarithmic bins: rain 0.1 µm–20 mm and snow/graupel
0.1 µm–50 mm. These finite bounds can omit tails of extreme distributions.
Wet snow retains WRF's 50 bins from 0.1–20 mm. Scattering tables are cached by
species/wavelength (maximum eight small entries); PSD work uses chunks of 256
cells, avoiding a whole-grid × size-bin allocation.

Seven independently published
[MIEV0 backscatter benchmarks](https://miepython.readthedocs.io/en/3.0.1/09_backscattering.html)
cover nonabsorbing, waterlike, absorbing and high-index spheres; tests use their
printed precision. A small-particle analytic Rayleigh limit and a large-drop
non-Rayleigh case check units and behavior. These qualify the scattering
calculation; bulk C/X output has **not** been validated against observed radar
or a published Thompson mixed-phase volume-scattering dataset.

## Column products

Names end in `_s`, `_c` or `_x` for the selected band.

| Variable | Units | Definition |
|---|---|---|
| `radar_dbz_*` | dBZ | Equivalent intrinsic reflectivity on WRF mass levels |
| `radar_composite_*` | dBZ | Maximum over the sampled vertical column |
| `radar_echo_top_*` | m AGL | Highest 18 dBZ crossing, or selected threshold; linearly interpolate in dBZ between mass levels |
| `radar_echo_top_status_*` | 1 | 0=no echo (height=0), 1=resolved crossing, 2=echo reaches model top (height is a lower bound) |
| `radar_vil_*` | kg/m² | Radar liquid-equivalent proxy from Greene–Clark, not actual condensate mass |

Heights use `(PH+PHB)/9.81` averaged from adjacent staggered interfaces, minus
`HGT`. Heights must increase and be nonnegative. VIL uses the
[NWS Greene–Clark discrete formula](https://www.weather.gov/ohx/vildensity2):
`3.44e-6 * sum(((Zi+Zi+1)/2)^(4/7) * delta_z_m)` in linear mm⁶/m³ reflectivity.
Method-floor cells contribute zero; values above 56 dBZ are capped, following
[NOAA DVIL](https://vlab.noaa.gov/documents/96675/1417806/DVILReferenceGuide.pdf/f4ee17a4-a9cf-441d-8fed-6fa65119ac14).
No surface or above-top extrapolation is made; the integral spans only the
lowest through highest mass levels. VIL retains contributions from ice as an
equivalent-liquid radar proxy. Finite inputs and outputs are required throughout.

All work is offline NumPy/xarray (optional miepython). It adds no timestep
kernel launches, host/device transfers, forecast allocations or model compile
cost. CLI optional dependency and missing-input errors occur before product
publication; the source files are opened read-only.
