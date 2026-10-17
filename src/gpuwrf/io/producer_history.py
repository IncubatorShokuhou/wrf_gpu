"""Producer outputs and narrowly qualified WRF-inactive history constants.

Class B admission: manager 2026-10-14 19:05Z; all-cell/all-25-frame bit proof:
proofs/t_writer_fields/INACTIVE_CPU_ALL_FRAMES.json. Constants here are WRF
Registry/reset values in the specified inactive branches, not missing physics.
"""
import numpy as np

CCN_FIELDS = tuple(f"CCN{i}_GS" for i in range(1, 8)) + ("EFCG", "EFIG", "EFSG", "WACT")
ICE_TYPES = ("ICE_TYPE", "ICE_TYPE2", "ICE_TYPE3")
STOCH_FIELDS = ("FIELD_CONV", "FIELD_PBL", "FIELD_SF")
PROFILE_FIELDS = ("RIMI", "EXCH_TKE")
FASDAS_FIELDS = ("HFX_BOTH", "QFX_BOTH")
# BEP+BEM (sf_urban_physics=3) green-roof/PV history (Registry.EM_COMMON:969-976, 'rh').
# The port refuses GR_FLAG/GR_FRAC_ROOF/PV_FRAC_ROOF != 0 at load (urban_bem_static.bem_class),
# so WRF's computation reduces to: urban_var_init EP_PV=TGR=DRAINGR=0 on every column
# (module_sf_urban.F:2935-2949) and bep_bem resets them to 0 on urban columns
# (module_sf_bep_bem.F:1055-1060); EP_PV then adds meso_urb_ac*d_urb*eppv with
# eppv = enerpv*pv_frac_roof*bl*bw = +-0 (module_sf_bem.F:903), leaving +0. QGR_URB4D is
# the held QGR_URB3D carry (urban_multilayer_hook.UrbanMultilayerCarry.green_roof).
# Original CPU proof: c57, every cell of all 25 frames (proofs/t_classd/URBAN_CPU_ALL_FRAMES.json).
URBAN_ZERO_FIELDS = ("TGR_URB4D", "DRAINGR_URB3D", "EP_PV_URB3D")
URBAN_FIELDS = ("QGR_URB4D", *URBAN_ZERO_FIELDS)
# CAM-MG (mp_physics=11) DLF2 (Registry/registry.cam:73, io rh, kg/kg/s), allocated by cammgmpscheme
# (Registry.EM_COMMON:3031). Its only writer is the UW shallow-convection driver
# (module_shcu_camuwshcu_driver.F:400, reached only under shcu_physics==CAMUWSHCUSCHEME,
# module_shallowcu_driver.F:545); CAMMGMP reads it intent(in) (module_mp_cammgmp_driver.F:184).
# Otherwise WRF keeps the allocation value initial_data_value=0. (inc/allocs_25.F,
# frame/wrf_num_bytes_between.c). The port refuses shcu_physics!=0 (io/namelist_binding.py).
# Original CPU proof: c89, every cell of all 25 frames +0 (proofs/t_cammgws/DLF2_CPU_ALL_FRAMES.json).
CAMMG_ZERO_FIELDS = ("DLF2",)
FIELDS = (*CCN_FIELDS, *ICE_TYPES, *STOCH_FIELDS, *PROFILE_FIELDS, "REFD_MAX", *FASDAS_FIELDS, *URBAN_FIELDS,
          *CAMMG_ZERO_FIELDS)


def inactive_names(namelist, lookup):
    """Use resolved model controls; an active branch never receives a constant."""
    mp = int(lookup(namelist, "mp_physics", 8))
    names = []
    if mp == 40 and int(lookup(namelist, "aercu_opt", 0)) == 0:
        names.extend(CCN_FIELDS)
    # Pristine Jensen only initializes/resets the classification arrays.
    if mp == 55:
        names.extend(ICE_TYPES)
    if mp in (17, 18, 19, 21, 22) and int(lookup(namelist, "nwp_diagnostics", 0)) == 0:
        names.append("REFD_MAX")
    if mp == 55 and int(lookup(namelist, "do_radar_ref", 0) or 0) == 1:
        # check_a_mundo: compute_radar_ref=1 allocates radar_refl (refl_10cm,refd_max);
        # only NWP diagnostics (refused with Jensen) update refd_max -> allocator zero.
        names.append("REFD_MAX")
    # t-classd: WRF-assigned constants of the admitted aerosol/radar branches.
    # Original CPU proof: c19/c65/c20, every cell of all 25 frames
    # (proofs/t_classd/CONSTANT_CPU_ALL_FRAMES.json).
    if mp in (28, 38):
        if int(lookup(namelist, "wif_input_opt", 0) or 0) != 2:
            # module_mp_thompson.F:1336-1338 nbca(i,k,j)=0 every call; the
            # loader refuses nonzero QNBCA input (nesting/scalar_initialization).
            names.append("QNBCA")
        if int(lookup(namelist, "aer_opt", 0) or 0) != 3:
            # module_radiation_driver.F:1653-1689: gt_aod only under aer_opt=3.
            names.append("TAOD5502D")
    if mp == 11 and int(lookup(namelist, "shcu_physics", 0) or 0) != 2:
        # No UW shallow convection (the only DLF2 writer): WRF writes its allocation zero.
        names.extend(CAMMG_ZERO_FIELDS)
    if mp == 40 and int(lookup(namelist, "do_radar_ref", 0) or 0) == 0:
        # module_mp_morr_two_moment_aero.F:1183-1203: diagflag .and. do_radar_ref==1 only.
        names.append("MSKF_REFL_10CM")
    options = lookup(lookup(namelist, "data_assimilation"), "options")
    surface_code = int(lookup(options, "surface_code", lookup(namelist, "grid_sfdda", 0)))
    if surface_code == 2 and lookup(namelist, "sf_surface_physics") == 4:
        # Manager approval 2026-10-14 21:59Z: surface_driver:2900/2904 runs
        # only inside classic-Noah CASE2. Noah-MP CASE4 retains Registry0.
        # Both fields: every cell/all25 original CPU frames in c78, sealed
        # FASDAS_NOAHMP_CPU_ALL_FRAMES.json. c53 reference stays incomplete.
        names.extend(FASDAS_FIELDS)
    if int(lookup(namelist, "sf_urban_physics", 0) or 0) == 3:
        names.extend(URBAN_ZERO_FIELDS)
    config = lookup(namelist, "stochastic_config")
    if (config is not None and int(lookup(namelist, "cu_physics", 0)) == 1
            and int(lookup(namelist, "bl_pbl_physics", 5)) == 5
            and lookup(namelist, "sf_surface_physics") == 4):
        for name, pattern in zip(STOCH_FIELDS, config.patterns[4:7], strict=True):
            if pattern is not None:
                names.append(name)
    return tuple(names)


def add_fields(fields, *, state, namelist, diagnostics, dimensions, requested_names,
               lookup, coerce, specs, shape_for_dimensions):
    def wanted(name):
        return requested_names is None or name in requested_names

    zeros = {}
    for name in inactive_names(namelist, lookup):
        if not wanted(name):
            continue
        shape = shape_for_dimensions(specs[name].dimensions, dimensions)
        supplied = lookup(diagnostics, name)
        if supplied is not None and np.any(np.asarray(supplied) != np.float32(0)):
            raise ValueError(f"{name} contradicts the qualified WRF-inactive constant")
        if shape not in zeros:
            zeros[shape] = np.zeros(shape, np.float32)
            zeros[shape].flags.writeable = False
        fields[name] = zeros[shape]
    profiles = (("RIMI",) if int(lookup(namelist, "mp_physics", 8)) == 13 else ()) + (
        ("EXCH_TKE",) if int(lookup(namelist, "bl_pbl_physics", 5)) == 12 else ())
    if (int(lookup(namelist, "ra_lw_physics", 4)), int(lookup(namelist, "ra_sw_physics", 4))) == (5, 5):
        profiles += tuple(f"RE_{species}_GSFC" for species in ("CLOUD", "ICE", "SNOW", "RAIN", "GRAUPEL", "HAIL"))
    if int(lookup(namelist, "sf_urban_physics", 0) or 0) == 3:
        profiles += ("QGR_URB4D",)
    for name in profiles:
        if not wanted(name):
            continue
        value = lookup(diagnostics, name)
        if value is None:
            value = lookup(state, name.lower())
        if value is None:
            # Direct writer API callers may provide a partial history stream.
            # Operational carries/restarts must provide these held outputs.
            continue
        if not np.isfinite(np.asarray(value)).all():
            raise ValueError(f"nonfinite held producer history {name}")
        shape = shape_for_dimensions(specs[name].dimensions, dimensions)
        fields[name] = coerce(name, value, shape, dtype=np.float32)
