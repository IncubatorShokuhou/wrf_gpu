"""Pure-Python WRF &fdda observation-nudging binding (no JAX import)."""
import math

DEFAULTS={
 'obs_nudge_opt':0,'max_obs':0,'fdda_start':0.,'fdda_end':0.,
 'obs_nudge_wind':0,'obs_nudge_temp':0,'obs_nudge_mois':0,'obs_nudge_pstr':0,
 'obs_coef_wind':0.,'obs_coef_temp':0.,'obs_coef_mois':0.,'obs_coef_pstr':0.,
 'obs_no_pbl_nudge_uv':0,'obs_no_pbl_nudge_t':0,'obs_no_pbl_nudge_q':0,
 'obs_sfc_scheme_horiz':0,'obs_sfc_scheme_vert':0,'obs_max_sndng_gap':20.,
 'obs_nudgezfullmin':50.,'obs_nudgezrampmin':50.,'obs_nudgezmax':3000.,
 'obs_sfcfact':1.,'obs_sfcfacr':1.,'obs_dpsmx':7.5,'obs_rinxy':0.,'obs_rinsig':0.,
 'obs_twindo':0.,'obs_npfi':0,'obs_ionf':1,'obs_idynin':0,'obs_dtramp':0.,
 'obs_prt_max':1000,'obs_prt_freq':1000,'obs_ipf_in4dob':False,'obs_ipf_errob':False,
 'obs_ipf_nudob':False,'obs_ipf_init':True,'obs_scl_neg_qv_innov':0,
}
for field in ('uv','t','q'):
 for regime in (1,2,4):
  DEFAULTS[f'obs_nudgezfullr{regime}_{field}']=-5000. if regime==4 else 50.
  DEFAULTS[f'obs_nudgezrampr{regime}_{field}']=50.
DOMAIN_KEYS={'obs_nudge_opt','fdda_start','fdda_end','obs_nudge_wind','obs_nudge_temp','obs_nudge_mois',
 'obs_nudge_pstr','obs_coef_wind','obs_coef_temp','obs_coef_mois','obs_coef_pstr',
 'obs_no_pbl_nudge_uv','obs_no_pbl_nudge_t','obs_no_pbl_nudge_q','obs_rinxy','obs_twindo','obs_ionf','obs_prt_freq'}


def observation_options(nml,domain):
    index=int(domain[1:])-1;values={}
    raw=nml.get('fdda') or {}
    for key,default in DEFAULTS.items():
        value=raw.get(key,default)
        k=index if key in DOMAIN_KEYS else 0
        if isinstance(value,(list,tuple)):value=value[k] if k<len(value) else default
        elif k>0:value=default
        values[key]=value
    return values


def observation_namelist_errors(nml,domain):
    o=observation_options(nml,domain)
    errors=[]
    try:
        value=float(o['obs_nudge_opt'])
        if not math.isfinite(value) or value!=int(value):raise ValueError
        code=int(value)
    except (TypeError,ValueError,OverflowError):return [f'{domain}: obs_nudge_opt must be 0 or 1']
    if code==0:return errors
    if code!=1:return [f'{domain}: obs_nudge_opt={code} is not WRF observation nudging (1)']
    if domain!='d01':errors.append(f'{domain}: observation nudging currently requires the root domain (d01)')
    physics=nml.get('physics') or {}
    for key in ('bl_pbl_physics','sf_sfclay_physics'):
        value=physics.get(key,5)
        if isinstance(value,(list,tuple)):value=value[0]
        if value!=5:errors.append(f'{domain}: observation nudging currently requires {key}=5 for qualified KPBL/surface ratios')
    surface=physics.get('sf_surface_physics',4)
    if isinstance(surface,(list,tuple)):surface=surface[0]
    if surface==3:errors.append(f'{domain}: observation nudging with RUC requires unported surface-ratio capture')
    for key in ('obs_ipf_in4dob','obs_ipf_errob','obs_ipf_nudob'):
        if o[key]:errors.append(f'{domain}: {key}=true requests unported per-step print diagnostics; set it false')
    binary=('obs_nudge_wind','obs_nudge_temp','obs_nudge_mois','obs_nudge_pstr','obs_no_pbl_nudge_uv',
      'obs_no_pbl_nudge_t','obs_no_pbl_nudge_q','obs_sfc_scheme_horiz','obs_sfc_scheme_vert','obs_idynin','obs_scl_neg_qv_innov')
    for key,value in o.items():
        if isinstance(DEFAULTS[key],bool):continue
        try:
            f=float(value)
            if not math.isfinite(f):raise ValueError
            if isinstance(DEFAULTS[key],int) and f!=int(f):raise ValueError
        except (ValueError,TypeError,OverflowError):
            errors.append(f'{domain}: &fdda {key} must be finite'+(' integer' if isinstance(DEFAULTS[key],int) else ''))
            continue
        if key in binary and f not in (0,1):errors.append(f'{domain}: &fdda {key} must be 0 or 1')
    try:
        for key in ('obs_rinxy','obs_twindo','obs_sfcfact','obs_sfcfacr','obs_dpsmx','obs_nudgezmax','obs_ionf'):
            if float(o[key])<=0:errors.append(f'{domain}: &fdda {key} must be positive for observation nudging')
        if float(o['obs_dpsmx'])>10:errors.append(f'{domain}: WRF observation nudging requires obs_dpsmx<=10 cb')
        for key in ('obs_nudgezfullmin','obs_nudgezrampmin','obs_max_sndng_gap','max_obs'):
            if float(o[key])<0:errors.append(f'{domain}: &fdda {key} must be nonnegative')
        if float(o['obs_nudgezfullmin'])+float(o['obs_nudgezrampmin'])>float(o['obs_nudgezmax']):
            errors.append(f'{domain}: WRF requires obs_nudgezfullmin+obs_nudgezrampmin<=obs_nudgezmax')
        if int(o['obs_idynin'])==1 and float(o['obs_dtramp'])==0:
            errors.append(f'{domain}: obs_idynin=1 needs a nonzero obs_dtramp')
        for field in ('wind','temp','mois'):
            if float(o[f'obs_coef_{field}'])<0:errors.append(f'{domain}: obs_coef_{field} must be nonnegative')
    except (ValueError,TypeError,OverflowError):pass  # numeric validation above owns errors
    return errors
