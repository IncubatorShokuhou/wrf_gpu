"""Pure configuration constants for the opt-in cuplus family."""
OPTION_NAMES=('cup_num_bins','cup_th_bin_size','cup_r_bin_size','cup_min_deep_freq',
              'cup_min_shallow_freq','cu_rad_feedback','icloud','cu_diag','kfeta_trigger',
              'shcu_aerosols_opt','shallowcu_forced_ra')
OPTION_DEFAULTS=(1,1.,1.,1.,1.,False,1,0,1,0,False)


def selection_errors(config):
    """Pre-JAX counterpart of the resolved profile check for both CLI loaders."""
    from collections.abc import Mapping
    from types import SimpleNamespace
    if not isinstance(config,Mapping):return resolved_errors(config) if hasattr(config,'cu_physics') else []
    from gpuwrf.io.namelist_binding import namelist_value
    from gpuwrf.io.wrf_registry_defaults import WRF_REGISTRY_DEFAULTS
    physics=config.get('physics',config);selected=physics.get('cu_physics',0)
    selected=selected if isinstance(selected,(list,tuple)) else [selected]
    errors=[]
    for index,cu in enumerate(selected):
        if int(cu) not in (7,10,11):continue
        domain=f'd{index+1:02d}'
        def get(key):
            value=namelist_value(config,key,domain)
            return WRF_REGISTRY_DEFAULTS[key][3] if value is None else value
        values=dict(zip(OPTION_NAMES,OPTION_DEFAULTS))
        for key in ('cu_rad_feedback','icloud','cu_diag','kfeta_trigger','shcu_aerosols_opt','shallowcu_forced_ra'):
            values[key]=get(key)
        for key,field in (('numbins','cup_num_bins'),('thbinsize','cup_th_bin_size'),('rbinsize','cup_r_bin_size'),
                          ('mindeepfreq','cup_min_deep_freq'),('minshallowfreq','cup_min_shallow_freq')):values[field]=get(key)
        values.update({key:get(key) for key in ('mp_physics','bl_pbl_physics','sf_sfclay_physics','ra_lw_physics','ra_sw_physics','km_opt')})
        values.update(cu_physics=int(cu),run_physics=True,rad_rk_tendf=1,use_noahmp=get('sf_surface_physics')==4)
        if int(cu)==11:values.update(cu_rad_feedback=True,icloud=1)
        if int(cu)==10:values.update(cu_diag=1)
        errors.extend(f'{domain}: {reason}' for reason in resolved_errors(SimpleNamespace(**values)))
    return errors


def resolved_errors(namelist):
    """Scientific profile limits of the composed CPU-qualified family."""
    import math
    cu=int(namelist.cu_physics)
    if cu not in (7,10,11):return []
    errors=[]
    if int(namelist.mp_physics)!=8 and not (cu==7 and int(namelist.mp_physics)==11):
        errors.append('cuplus currently qualifies Thompson mp_physics=8; other microphysics number/ice flags and unit pairings are unqualified')
    bl=int(namelist.bl_pbl_physics);sf=int(namelist.sf_sfclay_physics)
    if cu in (10,11) and (bl,sf)!=(5,5):
        errors.append('MSKF/KF-CuP currently require MYNN PBL/surface 5/5 for qualified ZOL/KPBL producers')
    if cu==7 and (bl,sf) not in ((9,5),(2,2)):
        errors.append('CAM-ZM requires the qualified CAM-UW/MYNN surface 9/5 or MYJ/Janjic 2/2 producer pairing')
    if cu==7 and bl==2 and bool(namelist.use_noahmp):
        errors.append('CAM-ZM/MYJ needs actual Janjic surface fluxes; retained Noah-MP adapter selects MYNN-SL, so this composite is unqualified')
    if (int(namelist.ra_sw_physics),int(namelist.ra_lw_physics))!=(4,4):
        errors.append('cuplus currently requires RRTMG SW/LW 4/4; other radiation cloud caller paths are unqualified')
    if int(namelist.kfeta_trigger)!=1:
        errors.append('cuplus supports original KF trigger 1; trigger 2 horizontal averaging is unported')
    if int(namelist.shcu_aerosols_opt)!=0:
        errors.append('KF-CuP aerosol/chemistry inputs are unported; use shcu_aerosols_opt=0')
    if int(namelist.icloud)!=1:
        errors.append('cuplus cloud mediation currently qualifies original icloud=1 only')
    if not bool(namelist.run_physics) or int(namelist.rad_rk_tendf)==0:
        errors.append('cuplus requires active physics with WRF RK source coupling (rad_rk_tendf=1)')
    if int(namelist.km_opt)==5:
        errors.append('MSKF/KF-CuP/CAM-ZM with SMS km_opt=5 is not a qualified composite profile')
    if cu==10:
        bins=namelist.cup_num_bins
        if isinstance(bins,bool) or int(bins)!=bins or int(bins)<1 or int(bins)%2!=1:
            errors.append('KF-CuP currently qualifies positive odd numBins only')
        for name in ('cup_th_bin_size','cup_r_bin_size'):
            value=float(getattr(namelist,name))
            if not math.isfinite(value) or value<=0:errors.append(f'{name} must be finite and positive')
        for name in ('cup_min_deep_freq','cup_min_shallow_freq'):
            value=float(getattr(namelist,name))
            if not math.isfinite(value) or not 0<=value<=1:errors.append(f'{name} must lie in [0,1]')
        if bool(namelist.cu_rad_feedback):
            errors.append('CuP uses its original automatic cloud mediation; additional Grell cu_rad_feedback diagnostics are unqualified')
    return errors
