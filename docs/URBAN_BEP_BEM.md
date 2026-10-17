# Multi-layer urban canopy and building energy

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

The qualified morphology is the default `URBPARM.TBL` medium-density urban
class (WRF's UTYPE=2). LCZ/WUDAPT, NUDAPT morphology, photovoltaic and green-roof
extensions are refused. Moving nests have no BEP/BEM qualification claim.
With those extensions off, BEM history writes WRF's `QGR_URB4D` (held
Noah-MP-initialized top-layer soil moisture, reset to 0 on urban columns) and
+0 `TGR_URB4D`/`DRAINGR_URB3D`/`EP_PV_URB3D`, bit-identical to original CPU-WRF
(proofs/t_classd/URBAN_CPU_ALL_FRAMES.json); checkpoints carry `QGR_URB3D`.

Qualification is against pristine CPU-WRF full caller and PBL savepoints in
both REAL4 and REAL8, including day/night, active multi-level exchange, and AC
on/off. The fixed, independently derived ROUND01 operand budget applies only
to conditioned REAL4 fluxes/tendencies. State, geometry, REAL8 and project D6
limits are unchanged. See [the numerical evidence](../proofs/v040/urban_bep_bem/README.md).

CPU coupled smoke uses an authentic 42×42 Madrid case with 44 levels and 31
urban cells. GPU/coupled-forecast qualification pending: CPU Pallas interpretation
does not establish GPU fidelity, E41, speed or launch counts. Urban material
arrays are compact over urban columns; six held surface-layer arrays cover the
domain. Default urban 0/1 programs retain their existing graph.
The smoke emulates the native CUDA target on CPU; portable-backend qualification
is separate. Full production graph and initial carry byte proofs bind the
executed smoke to the final composed tree.
