# Terrain shading and slope radiation

WRF has two independent switches. `slope_rad=1` adjusts surface shortwave for
terrain slope and aspect. `topo_shading=1` supplies the terrain shadow mask.
Both must be enabled for shadows to change the surface forcing. With
`slope_rad=0`, enabling `topo_shading` alone has no surface-flux effect.
There is no WRF `slope_rad=2` mode.

For a three-domain namelist using the RRTMG pair:

```fortran
&physics
 ra_lw_physics = 4, 4, 4,
 ra_sw_physics = 4, 4, 4,
 radt = 30, 30, 30,
 slope_rad = 1, 1, 1,
 topo_shading = 1, 1, 1,
 shadlen = 25000.,
/
```

Run the nests together as described in the [nesting guide](nesting.html).
The input supplies real latitude, longitude, terrain and map rotation.
`shadlen` sets WRF's directional search extent of `INT(shadlen/dx + 1)` grid
steps (default 25000 m); it is not a circular terrain-padding radius. Radiation and its diagnostics are held at radiation
cadence. SWDOWN denotes horizontal downward flux; SWNORM is the slope/shadow
adjusted surface forcing.

The pending v0.4.4 correction fixes two differences from WRF. First, WRF's
shadow search uses its own solar clock without the equation of time, while
RRTMG and the slope correction use the full clock; the port now keeps the two
apart. Second, WRF keeps each nest's preceding shadow height at its outer two
rows; the child does not receive freshly computed parent heights on every
radiation call. The height uses one optional fp32 plane per domain, kept with
the model state between calls. Shading-enabled checkpoints written before this plane existed are
refused; they cannot reconstruct its history. Shading-off runs keep it absent. Goddard and FLG/GFDL radiation do not use
this plane.

**Cases that already use `slope_rad=1` and `topo_shading=1` get different
results with v0.4.4.** This is an intentional, WRF-faithful correction, so
earlier forecasts of such cases are not byte-comparable. The shading correction is inactive when shading is off; other v0.4.4
initialization/clock changes still apply, so this is not whole-program identity
with an earlier release.
Every nest starts from WRF's allocator-zero height; the Noah-MP initial
radiation call does not count as a shadow history call.

CPU-WRF under MPI propagates shadows across patch boundaries through halo
iterations. The GPU port runs one patch per domain, so small differences along
CPU-WRF patch boundaries are expected decomposition behaviour, not a model error.

The composed release's forecast gate is reported in the [release notes](../release_notes/RELEASE_NOTES_v0.4.4.md); operator-specific native GPU/original-WRF comparison remains separate. CPU comparisons
against pristine WRF pass on the real 3 km and 1 km Tenerife nest terrain over
72 hours: all 290 shadow masks match exactly, and the largest shadow-height
difference is about 0.005 m. No feature-isolated GPU speed, memory-peak or forecast-error
result is claimed. The held planes of a three-nest Tenerife case take about
200 kB (from the array shapes). The intended target is nested runs with RRTMG
longwave and shortwave (`ra_lw_physics = ra_sw_physics = 4`).

## Compare with original CPU-WRF

Keep the same terrain, radiation cadence, `shadlen` and start time, and include
daylight hours: a short night-time run cannot show shading effects. SWDOWN,
SWNORM, COSZEN and the surface fluxes are already in WRF's default history. The
shadow height and the direct/diffuse split are not; request them in
`iofields_filename`:

```text
+:h:0:HGT_SHAD,SWDDIR,SWDDIF
```

Compare like with like: SWDOWN is horizontal, SWNORM is slope- and
shadow-adjusted. The [evidence report](../proofs/w44_topo/TOPO.md) gives the
exact scope, tolerances and pending gates.
