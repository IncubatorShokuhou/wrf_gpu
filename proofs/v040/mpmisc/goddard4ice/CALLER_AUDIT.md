# Pristine WRF mp=7 caller and constants audit

Reference: WRF 4.7.1 `module_mp_gsfcgce_4ice_nuwrf.F`, SHA256
`79b782f6832ce45c0a13f463841d1820cabc4de0049ac60b5c2a23d24389cb8b`.
`module_microphysics_driver.F:2351–2426` calls the public original wrapper.

The public prognostics are dry potential temperature TH and seven mixing
ratios QV/QL/QR/QI/QS/QG/QH. Graupel and hail are separate advected categories.
Inputs RHO, PII, P, DT_IN, geometric mass-level Z, terrain HT, DZ8W, W, XLAND and
DX are retained exactly. W is the BOTTOM staggered w face (solve_em.F:3904),
not a mass-level average. Z/HT control sedimentation distance, DZ8W its mass
flux divergence. Sedimentation uses sqrt(1.29/rho); saturation uses
sqrt(rho(k=1)/rho(k)). These reference densities intentionally differ.

The wrapper hardcodes itaobraun=0, improve=8, new_ice_sat=9 and id=0; its call
has no active ihail/ice2 switches. WRF-Chem aerosol coupling is excluded from
this implementation. Non-Chem CCN is fixed at 1500 cm^-3 over land (XLAND=1),
150 cm^-3 over water (XLAND=2). No XICE input exists. The six gamma_toshi
REAL returns for these exact CCN arguments were computed by the unmodified
original DOUBLE routine; 224 constant entries, including 10 REAL tables, are likewise extracted
by a separate getter containing the exact original routine bodies. Forecast
arithmetic stays REAL4; there is no new FP64 hot-path island.

ACPHYSC/E/D/S/M/F resets on ITIMESTEP=1, then accumulates potential-temperature
latent budgets. Registry flags are r; six RE_*_GSFC are rh (microns). The radiation driver (1139–1157) copies RE_CLOUD/ICE/SNOW_GSFC times
1.e-6 to general metre-valued radii for mp7; physics_init (988–1024) enables
those RRTMG/Goddard optics when use_mp_re=1. GSFC radii start at zero
(physics_init:1302–1316). The
caller derives ACPHYS_TOT by left-associated c+e+d+s+m+f; it never passes that
sum back. Instant PHYSC/E/D/S/M/F and five PREC*3D have Registry '-' flags,
so native forecasts elide those outputs; full=True retains them for oracles.

RAINNCV includes rain+snow+graupel+cloud-ice+hail. SNOWNCV includes snow only;
SR includes all four frozen species. Noah-MP's original surface caller passes
RAINNCV as total and separate SNOWNCV/GRAUPELNCV/HAILNCV explicit channels;
cloud ice is consequently retained in the total channel, not SNOWNCV. The
extra unheld rain_other output is exactly pptrain+pptice, computed before
any frozen-category subtraction; this prevents negative cancellation in
the disjoint internal rain accumulator. Original RAINNCV total association
remains unchanged and is available separately for the Noah caller.

Original physics bodies and reference fixtures are never patched. Builds use
compiler zero initialization, integer=0/logical=false and production FP traps
off. Six complete columns are BYTE-identical with alternate snan initialization
for all 47 dumped columns; no active undefined-read sensitivity was observed.

Preregistered holdouts: all six regimes at dt18/dx9000/water/3 own steps and
dt54/dx3000/land/2 steps, with cumulative latent budgets carried between calls.
All 18 columns compare all 36 public output groups (8 prognostics, 6 radii,
12 instantaneous/cumulative budgets, 5 fluxes and 5 surface diagnostics).
The independent compiled envelope sums each of 14 input vectors' maximum
+/-1ULP original-Fortran response, plus the original physics-only native-FMA
spread. The generic driver is never built with native FMA; its 14 inputs must
remain BYTE equal. Full 24h coupled D6 and real-GPU E41 remain release gates.

Performance: one Pallas program owns each column's five adaptive Courant loops
and downward sweeps. It pads only its register column to a power of two; no
full-domain intermediates or transpose are introduced. Existing qg/qh output
buffers supply two neighbor reloads separated by one CTA barrier. Default
native outputs retain only 20 scientific column fields and 6 surface fields;
source-only instantaneous/flux outputs are elided. Measured GPU speed/VRAM
are unknown; CPU interpretation and CUDA IR lowering are separately recorded.
