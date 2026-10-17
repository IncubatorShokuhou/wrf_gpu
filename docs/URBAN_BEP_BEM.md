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
class (WRF's UTYPE=2). LCZ/WUDAPT with BEP/BEM is not supported, and neither are
NUDAPT morphology, photovoltaic and green-roof extensions. LCZ land use is
available only with the single-layer urban model (`sf_urban_physics=1`); see
[LCZ, urban morphology and anthropogenic heat](urban_lcz.md). Moving nests are
not validated with BEP/BEM.
With those extensions off, BEM history writes WRF's `QGR_URB4D` (held
Noah-MP-initialized top-layer soil moisture, reset to 0 on urban columns) and
+0 `TGR_URB4D`/`DRAINGR_URB3D`/`EP_PV_URB3D`, bit-identical to original CPU-WRF
(proofs/t_classd/URBAN_CPU_ALL_FRAMES.json); checkpoints carry `QGR_URB3D`.

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
