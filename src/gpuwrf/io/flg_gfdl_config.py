"""Backend-dark admission of experimental CPU-only FLG/GFDL radiation."""
from collections.abc import Mapping
from gpuwrf.backends.platform import target_backend


def selection_errors(options,*,backend=None):
    source=options.get('physics',options) if isinstance(options,Mapping) else options
    def get(key,default):
        value=source.get(key,default) if isinstance(source,Mapping) else getattr(source,key,default)
        return default if value is None else value
    values={key:get(key,default) for key,default in dict(ra_lw_physics=4,ra_sw_physics=4,mp_physics=8,
        cu_physics=0,sf_surface_physics=4,sf_urban_physics=0,sf_lake_physics=0,
        topo_shading=0,slope_rad=0,swint_opt=0,aer_ra_feedback=0,icloud=1).items()}
    n=max([len(v) for v in values.values() if isinstance(v,(list,tuple))] or [1]);errors=[]
    for domain in range(n):
        c={k:int(v[min(domain,len(v)-1)] if isinstance(v,(list,tuple)) else v) for k,v in values.items()}
        selected=set((c['ra_lw_physics'],c['ra_sw_physics']))&{7,99}
        if not selected:continue
        if (backend or target_backend())!='cpu':
            errors.append(f'FLG/GFDL domain {domain+1}: EXPERIMENTAL CPU-ONLY; native GPU column kernel pending; set JAX_PLATFORMS=cpu')
        if len(selected)!=1 or c['ra_lw_physics']!=c['ra_sw_physics']:
            errors.append(f'FLG/GFDL domain {domain+1} requires paired ra_lw_physics=ra_sw_physics=7 or 99')
        required=dict(mp_physics=8,cu_physics=0,sf_surface_physics=4,sf_urban_physics=0,sf_lake_physics=0,
            topo_shading=0,slope_rad=0,swint_opt=0,aer_ra_feedback=0,icloud=1)
        bad=[f'{key}={value}' for key,value in required.items() if c[key]!=value]
        if bad:errors.append(f'FLG/GFDL domain {domain+1} qualified caller requires '+', '.join(bad)+
            '; convection interval producers/other MP and land pairings remain unqualified')
    return errors


def validate_selection(options,*,backend=None):
    errors=selection_errors(options,backend=backend)
    if errors:raise ValueError('; '.join(errors))


def bootstrap_refusal(argv):
    """CLI run refusal before gpuwrf's package bootstrap imports JAX."""
    if len(argv)<2 or argv[1]!='run' or target_backend()=='cpu':return None
    from pathlib import Path
    def option(name):
        for index,arg in enumerate(argv):
            if arg.startswith(name+'='):return arg.split('=',1)[1]
            if arg==name and index+1<len(argv):return argv[index+1]
        return None
    namelist=option('--namelist');input_directory=option('--input-dir')
    if namelist is None and input_directory is not None:namelist=str(Path(input_directory)/'namelist.input')
    if namelist is None or not Path(namelist).is_file():return None
    from gpuwrf.io.namelist_parser import parse_namelist
    try:config=parse_namelist(Path(namelist))
    except (ValueError,OSError):return None  # Ordinary CLI parsing handles malformed input.
    physics=config.get('physics',{})
    for key in ('ra_lw_physics','ra_sw_physics'):
        value=physics.get(key,4);values=value if isinstance(value,(list,tuple)) else (value,)
        if any(v in (7,99) for v in values):
            return 'FLG/GFDL are EXPERIMENTAL CPU-ONLY: native GPU column kernel pending; set JAX_PLATFORMS=cpu'
    return None
