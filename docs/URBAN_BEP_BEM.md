# Multi-layer urban canopy and building energy

**Status in v0.4.0: Validated 24 h.** Passed a 24-hour comparison with the original CPU WRF in the v0.4.0 release check, in one tested single-domain configuration. Other combinations and nested domains are not validated. Its compiled GPU code exceeds the project's code-quality limits for speed and robustness; it runs correctly, but its speed has not been measured.

The native single-domain and nested pipelines accept `sf_urban_physics=2`
(BEP) and `3` (BEP+BEM) with Noah-MP (`sf_surface_physics=4`), YSU or BouLac
(`bl_pbl_physics=1/8`), revised MM5 (`sf_sfclay_physics=1`), and RRTMG
(`ra_lw_physics=4`, `ra_sw_physics=4`). The legacy daily replay pipeline refuses
these options because it replays CPU land fields instead of evolving buildings.

WRF allows YSU, MYJ and BouLac with BEP/BEM. BEP+MYJ requires the separate
`module_bl_myjurb` implementation, which this port does not contain; it is refused.
Other PBL pairings are also refused before JAX initialization.

The implementation includes canyon radiation and shadowing, roof/wall/ground
heat diffusion, multi-layer momentum/heat/moisture/TKE sources, Noah-MP rural
fraction blending and the urban PBL consumers. BEM adds indoor radiation,
windows, floors, equipment/occupants, ventilation, heating/cooling and moisture
regulation. Surface diagnostics follow WRF's urban overwrite. Checkpoints retain
material/indoor/water state, urban PBL sources, held surface-layer outputs and
held direct/diffuse radiation; incomplete active urban checkpoints are refused.

The supported morphology is the default `URBPARM.TBL` medium-density urban
class (WRF's UTYPE=2). LCZ/WUDAPT with BEP/BEM is not supported, and neither is
NUDAPT morphology. LCZ land use is available only with the single-layer urban
model (`sf_urban_physics=1`); see
[LCZ, urban morphology and anthropogenic heat](urban_lcz.md). Moving nests are
not validated with BEP/BEM.

BEP+BEM (`sf_urban_physics=3`) runs WRF's rooftop extensions from `URBPARM.TBL`:

- **Photovoltaic roofs** (`PV_FRAC_ROOF>0`): panel energy balance and
  production (`hsfluxpv`), panel-shaded roof radiation (`radfluxspv`), the panel
  terms of the urban upward radiation and the panel sensible heat in the roof
  source. History `EP_PV_URB3D` (W m⁻²) is WRF's per-call building-weighted sum on
  urban columns; checkpoints carry `T_PV_URB3D`.
- **Green roofs** (`GR_FLAG=1` with `GR_FRAC_ROOF>0` for the class, `GR_TYPE`
  1 or 2): 10-layer soil heat and moisture with hydraulic properties
  (`roof_temp_veg`, `soil_moist`), vegetated-roof fluxes with stomatal resistance
  (`flux_flat_roof`), the green-roof coupling into the BEM roof (`wall_gr`) and
  hourly irrigation (`IRHO`). History `QGR_URB4D`, `TGR_URB4D` and
  `DRAINGR_URB3D` are WRF's per-call sums; checkpoints carry the soil
  temperature/moisture, fluxes and drainage. Checkpoints bind the active PV/green-roof
  selection; a checkpoint whose PV/green-roof state is incomplete, misshapen or from
  another selection is refused on save and read.

`GR_FRAC_ROOF>0` with `GR_FLAG=0` (WRF then blends roof temperatures with an
uninitialized green-roof layer), `GR_FLAG=1` without a fraction for the class and
other `GR_TYPE` values are refused at load.
With both extensions off, BEM history writes WRF's `QGR_URB4D` (held
Noah-MP-initialized top-layer soil moisture, reset to 0 on urban columns) and
+0 `TGR_URB4D`/`DRAINGR_URB3D`/`EP_PV_URB3D`, bit-identical to original CPU-WRF
(proofs/t_classd/URBAN_CPU_ALL_FRAMES.json); checkpoints carry `QGR_URB3D`.
The PV and green-roof branches are checked against pristine WRF column by column
(proofs/w41_writergaps/README.md), including a constructed green-soil/roof
temperature contrast that exercises `wall_gr`. A 9 h GPU run with both
extensions on finishes with finite output and writes all four fields every hour;
that shows the branches run, not that the forecast matches CPU-WRF, which has not
been compared yet. `DRAINGR_URB3D` is the drainage of each call (WRF resets it
every call), although its Registry description says accumulated.

Component checks compare against the original CPU WRF code (full caller and PBL
saved states) in both single (REAL4) and double (REAL8) precision, including
day/night, active multi-level exchange, and air conditioning on/off. A fixed,
independently derived rounding allowance applies only to conditioned
single-precision fluxes and tendencies. State, geometry, double-precision and
forecast comparison limits are unchanged. See [the numerical evidence](../proofs/v040/urban_bep_bem/README.md).

The short CPU test uses an authentic 42×42 Madrid case with 44 levels and 31
urban cells. Urban material arrays are compact over urban columns; six held
surface-layer arrays cover the domain. Runs with `sf_urban_physics=0` or `1`
keep their existing program.
