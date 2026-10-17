# Milbrandt-Yau two-moment microphysics

Select `mp_physics=9` for the experimental WRF Milbrandt-Yau scheme on a single
domain in the native driver initialized from `wrfinput`/`wrfbdy`. CPU-history
replay is refused by CLI preflight and the direct daily API because it cannot
retain the scheme's mandatory per-step radar diagnostic. Nested runs
remain refused until parent-to-child forcing is qualified. `km_opt=2` is also
refused for this scheme until its extra-scalar SGS diffusion is qualified.

CPU-oracle-qualified; GPU/coupled-forecast qualification pending. The port follows
pristine WRF's `module_mp_milbrandt2mom.F`: continental CCN type 2, all warm/cold
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
The optimized compiler comparison retains one primary ice-number exceedance:
0.104 #/kg at 2772 #/kg. It lies inside a separately derived one-ULP input
spread from 36 original-WRF arms; the registered relative bound is unchanged.
This follows the wave's compiler-conditioning policy, with forecast qualification pending.

The native kernel uses REAL32 arrays and literal RN32 operators. WRF's explicit
DBLE rain-intercept arithmetic remains DOUBLE; initialization gamma constants
are calculated on the host once. CPU qualification emulates scalar REAL
transcendental rounding with widened evaluation; this is not GPU performance or
native-executable fidelity evidence. GPU timing, register usage and peak VRAM
remain unmeasured. Existing release configurations are unchanged by this opt-in.
