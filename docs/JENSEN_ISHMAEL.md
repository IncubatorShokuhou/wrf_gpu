# Jensen-ISHMAEL (mp_physics=55)

**Status in v0.4.0: Validated 24 h.** Passed a 24-hour comparison with the original CPU WRF in the v0.4.0 release check, in one tested single-domain configuration. Other combinations and nested domains are not validated. Its compiled GPU code exceeds the project's code-quality limits for speed and robustness; it runs correctly, but its speed has not been measured.

Single-domain option. NTU and CAM-MG are separate ports.

The reference is the unchanged WRF 4.7.1 `module_mp_jensen_ishmael.F`, called
through its original public wrapper. The literal REAL4/REAL8 port matches all
34 reference columns byte for byte (50,456 public values). Two original public
steps evolve the ice habits and exercise 245 nonzero REAL4 aggregation mass
transfers, compared with zero on the first step. The compiler-defined zero
initialization reference and alternate initialization give identical public
outputs. See `proofs/s2_mpmore/jensen55/UPSTREAM_WRF_BUGS.md`.

The compiled arithmetic has known rounding deviations that fall outside the
spread produced by small input-rounding and fused multiply-add changes in the
original WRF code. They do not pass the strict column limits; the 24-hour
forecast comparison with the original CPU WRF is the deciding check. An earlier
interpreted probe of the GPU kernel reported 335 values outside that spread:
maximum temperature excess over its limit is 6.104e-5 K (maximum baseline
difference at these values is 1.526e-4 K), and maximum mass excess is
6.87e-9 kg/kg. The evidence lists the affected values, baseline differences and
limits separately. No general tolerance was increased. GPU speed has not been
measured.

The State carries vapor, cloud, rain, three ice masses and numbers, rain number,
and six ice volume/aspect moments. All three masses use the moisture transport
loop; all ten scalar moments use the scalar loop. Root flow boundaries and RK1
mixing include the extra categories and moments. Both input pipelines read
QICE2/QNICE2, QICE3/QNICE3 and QVOLI/QAOLI (categories 1–3), with original
Registry zero cold starts for absent fields. Selected restart files require
every habit moment and held cloud/ice radius and preserve them losslessly.

Full, operational and requested-subset history streams emit these represented
fields with original Registry metadata and refuse nonfinite values. Unused
snow/graupel, droplet and CCN slots are omitted. Cloud and ice radii mediate
RRTMG as in WRF (`has_reqc=1`, `has_reqi=1`, `has_reqs=0`); summed ice mass
uses the snow optics channel with its held ice radius. Fall speeds, diameters,
densities and aspect ratios are held from the original per-habit scheme
outputs, restarted with a missing-field check, and emitted as
`V_ICE*`, `D_ICE*`, `RHO_ICE*` and `PHI_ICE*`. See
[ice diagnostics](physics/ice_diagnostics.md). Ice type and radar diagnostics
remain omitted.

The GPU route is one Pallas program per column. Adaptive sedimentation and
vertical density carry remain inside that program; no domain-wide loop or
host work is added to the timestep. Only State replacements and precipitation
are stored, including the held ice diagnostics. The retained JAX loops serve
the CPU reference route. WRF's gammln is the sole permitted DOUBLE island;
retaining diagnostics may keep arithmetic that the older output projection
eliminated. GPU cost has not been measured. Other
arithmetic and optional carry are REAL4.

The 44,468,557-byte packaged LUT archive contains 51,737,636 raw host array
bytes. The actual cached arrays, caller maps and defaults enter the compiled-program
cache key; changed values reject old executables. Typed REAL8 tables are
reference-only temporary promotions, never a stale second cache. These host-side
numbers do not describe GPU peak memory, which has not been measured.

The final selected-table key probe measured 53.14 ms for its first hash and
48.49 ms for the warm median; this host work occurs outside timesteps. A
two-step CPU test on a Swiss case is finite with zero guard events; both native
and daily loaders accept 55. The default forecast program is unchanged.
