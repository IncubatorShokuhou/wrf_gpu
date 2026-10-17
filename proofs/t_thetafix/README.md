# Eight microphysics theta handoffs

Based on t-c03rain's final Lin fix `1612acc49`. The model change is confined to
mp=1/3/4/6/10/13/14/97 in `scan_adapters.py`: pass `_dry_theta_view(state)` to
the scheme, then restore theta_m with `_theta_m_from_microphysics`. Lin,
Thompson, WDM6, shared helpers, number/precipitation mapping and persistent State
fields are unchanged. `source_audit.json` records the eight changed functions
and 47 unchanged functions against the Lin base.

The fixture contains six **observed Swiss CPU-WRF columns**, 44 levels, from
`RD11/cpu/wrfout_d01_2023-01-15_06:00:00`. Columns maximize the six water species
on the interior. Mass, pressure, geopotential, winds and Thompson Ni/Nr are
observed; extra scheme numbers are explicitly initialized (see manifest).
Graupel is only a trace in this case; this is a caller handoff gate, not a claim
of new graupel-regime qualification.

`build_oracle.py` copies pristine WRF 4.7.1 scheme bodies and dependencies byte
for byte into a private build directory. Its logging/fatal stubs contain no
physics. `caller.f90.in` receives literal moist prep/finish statements extracted
from pristine `module_big_step_utilities_em.F`, calls each original scheme
wrapper with the microphysics-driver mapping, and evolves **two consecutive
calls** with dt=18 s. Exner and geometry use the documented port caller
convention; this does not qualify their separate known discrepancies versus
whole CPU-WRF (including density and Goddard terrain). It is not a forecast
trajectory or station gate.

Tests exercise three separate checks:

- The real adapter receives the original solver's dry TH_PHY and returns the
  original moist carry, advancing the previous call's State. Direct REAL4 and
  the existing fp64 dispatcher island are covered. Theta uses two actual
  REAL4 spacings (6.103515625e-5 K at 256–512 K), matching the independently
  compiled original kind spread. Other mapped leaves must be byte exact.
- The **actual compiled dispatcher and JAX scheme** execute both calls and
  compare thermal/mass/total precipitation to original WRF at existing
  Kessler absolute bands: 2e-3 K, 1e-7 kg/kg, 2e-4 mm. Existing isolated-solver
  machine-precision gates and forecast tolerances remain separate and unchanged.
- Replacing dry theta by State.theta_m at each of the eight scheme inputs
  produces genuine assertion failures, with setup errors counted separately.

The optional original REAL8 fixture preserves REAL4 carries at every call
boundary, matching the pre-existing production island ABI. **WDM5 Nn/Nc do
not meet the canonical 2% number band on these columns:** original REAL4 versus
REAL8 itself differs by 9.8852928e7 #/kg. This is retained as precision-sensitivity
evidence, not a number gate pass or waiver. Full GPU/D6 qualification remains
with v-prep/v-run.

Default Swiss and PROD d01 native-facade identities are f672a573 and 6fc715a2.
Thus the default compiled arithmetic and State memory schema are unchanged.
For affected options, the existing density path already calculates dry theta
in seven of the eight adapters; compilation can share that expression. Moist
finish adds elementwise arithmetic and a temporary thermal result, with no new
host transfer, synchronization, persistent field, cache provider or runtime
asset. Actual GPU launch/peak-memory effects are unmeasured; the already
scheduled composite reruns must check them. No GPU speed claim is made here.

Affected staged composites: c02(mp1), c04(mp3), c05(mp4), c06(mp6), c08(mp10),
c09(mp13), c10(mp14), c25/c61(mp97). c03(mp2) belongs to the Lin prerequisite.

Additional source audit (reported to the manager; not changed here): raw
theta_m is used for physical T in `_surface_layer_bottom_inputs`,
`_flux_handles_from_hfx_qfx`, `tiedtke_adapter`, `ntiedtke_adapter`, `gf_adapter`,
`bmj_adapter` and `_pbl_surface_forcing`. CU/PBL also require scheme-specific
dry-tendency conversion. KF/Grell use the established dry prep and moist
tendency helper. No land-surface solver is implemented in this file; those
hooks are separate. These are additional caller investigations, not covered
by this eight-MP fix.

Rebuild on lane cores, with JAX_PLATFORMS=cpu and the required QUIET guard:

```sh
python proofs/t_thetafix/build_oracle.py <USER_HOME>/wrf_gpu2_lanes/t-thetafix/build fp32
PYTHONPATH=src python proofs/t_thetafix/generate.py <USER_HOME>/wrf_gpu2_lanes/t-thetafix/build <USER_HOME>/wrf_gpu2_lanes/t-thetafix/fixture
```

For the transparency fixture, build with `fp64` into a separate private directory.
Delete build directories after the last dependent arm. The tracked manifests
bind all original sources, driver bytes, compiler flags, input file and fixtures.
