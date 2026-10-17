# Jensen-ISHMAEL (mp_physics=55)

Experimental, single-domain option. CPU-oracle-qualified; GPU and D6 coupled
forecast qualification are pending. NTU and CAM-MG are separate ports.

The reference is the unchanged WRF 4.7.1 `module_mp_jensen_ishmael.F`, called
through its original public wrapper. The literal REAL4/REAL8 port matches all
34 reference columns byte for byte (50,456 public values). Two original public
steps evolve the ice habits and exercise 245 nonzero REAL4 aggregation mass
transfers, compared with zero on the first step. The compiler-defined zero
initialization reference and alternate initialization give identical public
outputs. See `proofs/s2_mpmore/jensen55/UPSTREAM_WRF_BUGS.md`.

The compiled arithmetic has disclosed rounding deviations outside independent
original input-ULP/FMA controls. These are experimental deviations, not strict
enclosure passes; D6 versus CPU-WRF decides forecast qualification. The prior
native interpreter probe reports 335 values outside the independent enclosure:
maximum temperature excess over its bound is 6.104e-5 K (maximum baseline
difference at these values is 1.526e-4 K), and maximum mass excess is
6.87e-9 kg/kg. The final composed receipt records the affected values,
baseline differences and independent bounds separately. No general tolerance is
increased. GPU execution, resource/E41 qualification and performance remain
unmeasured.

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
outputs, restarted with an E78 missing-field guard, and emitted as
`V_ICE*`, `D_ICE*`, `RHO_ICE*` and `PHI_ICE*`. See
[ice diagnostics](physics/ice_diagnostics.md). Ice type and radar diagnostics
remain omitted.

The GPU route is one Pallas program per column. Adaptive sedimentation and
vertical density carry remain inside that program; no domain-wide loop or
host work is added to the timestep. Only State replacements and precipitation
are stored, including the held ice diagnostics. The retained JAX loops serve
the CPU reference route. WRF's gammln is the sole permitted DOUBLE island;
retaining diagnostics may keep arithmetic that the older output projection
eliminated. Actual GPU cost and resource qualification remain pending. Other
arithmetic and optional carry are REAL4.

The 44,468,557-byte packaged LUT archive contains 51,737,636 raw host array
bytes. The actual cached arrays, caller maps and defaults enter the AOT key;
changed-value regressions reject old executables. Typed REAL8 tables are
reference-only transient promotions, never a stale second cache. Asset memory
and host key cost are separate from unmeasured GPU peak memory.

The final selected-table key probe measured 53.14 ms for its first hash and
48.49 ms for the warm median; this host work occurs outside timesteps. The
actual Swiss two-step CPU smoke is finite with zero census guard events;
both native and daily loaders bind 55. Default program identities remain
Swiss `f672a573…` (approved glacier fix) and PROD d01 `6fc715a2…`.
