# Milbrandt-Yau two-moment microphysics

**Status in v0.4.0: Validated 24 h.** Passed a 24-hour comparison with the original CPU WRF in the v0.4.0 release check, in one tested single-domain configuration. Other combinations and nested domains are not validated. Its compiled GPU code exceeds the project's code-quality limits for speed and robustness; it runs correctly, but its speed has not been measured.

Select `mp_physics=9` for the WRF Milbrandt-Yau scheme on a single
domain in the native driver initialized from `wrfinput`/`wrfbdy`. CPU-history
replay is refused by CLI preflight and the direct daily API because it cannot
retain the scheme's mandatory per-step radar diagnostic. Nested runs
remain refused until parent-to-child forcing is validated. `km_opt=2` is also
refused for this scheme until its extra-scalar SGS diffusion is validated.

The port is also checked on the CPU against the original WRF code. It follows
WRF's `module_mp_milbrandt2mom.F`: continental CCN type 2, all warm/cold
processes, adaptive sedimentation, separate hail, six number mixing ratios, and
per-step radar reflectivity. The six number species are transported with the
scalar advection option, mixed and handled by root `flow_dep_bdy`. Restart retains
all six, hail mass/accumulation, and held `REFL_10CM`; incomplete active restarts
are refused.

WRF computes `Dm_*` as local diagnostic diameters and discards them in its
wrapper. Its effective-radius assignments are commented out, and physics
initialization leaves `has_reqc/i/s=0` for mp=9. This port likewise does not couple
these diameters to radiation. One unused `Dm_i` differs by 0.230 micrometres in
an adversarial oracle point with residual ice of order 1e-13 kg/kg; the original
diameter check and failure are retained. Literal REAL4 execution is byte-exact
against the unmodified no-FMA `-O0` oracle (3168 values, including these locals).
With compiler optimizations on, one ice-number value exceeds its limit:
0.104 #/kg at 2772 #/kg. It lies inside the spread that a one-unit rounding
change in the inputs produces across 36 runs of the original WRF code; the
registered relative limit is unchanged.

The native kernel uses REAL32 arrays and literal RN32 operators. WRF's explicit
DBLE rain-intercept arithmetic remains DOUBLE; initialization gamma constants
are calculated on the host once. The CPU checks emulate single-precision
math-function rounding with wider evaluation; they are not GPU performance
evidence. GPU timing, register usage and peak GPU memory have not been measured. Existing release configurations are unchanged by this opt-in.
