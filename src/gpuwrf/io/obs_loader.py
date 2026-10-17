"""Initialization-only OBS_DOMAIN loading for the operational pipelines."""
from datetime import datetime
from pathlib import Path
import hashlib,json
from gpuwrf.io.obs_namelist import observation_options,observation_namelist_errors


def load_observation_nudging(input_dir,wrf_namelist,domain,dt_s):
    options=observation_options(wrf_namelist,domain)
    if options['obs_nudge_opt']==0:return None
    errors=observation_namelist_errors(wrf_namelist,domain)
    if errors:raise ValueError('; '.join(errors))
    from gpuwrf.io.obs_domain import read_obs_domain
    from gpuwrf.io.obs_prepare import prepare_reports
    from gpuwrf.io.obs_projection import ObservationProjection
    from gpuwrf.io.netcdf_lock import Dataset
    import jax.numpy as jnp
    from gpuwrf.assimilation.obs_driver import ObservationInputs
    root=Path(input_dir)
    obsfile=root/f'OBS_DOMAIN{100+int(domain[1:]):03d}'
    with Dataset(root/f'wrfinput_{domain}') as ds:
        nx=len(ds.dimensions['west_east']);ny=len(ds.dimensions['south_north'])
        code=int(ds.MAP_PROJ)
        if code==6:raise ValueError('obs nudging latitude/longitude projection is awaiting pristine qualification')
        projection=ObservationProjection(code,float(ds['XLAT'][0,0,0]),float(ds['XLONG'][0,0,0]),
                     float(ds.TRUELAT1),float(ds.TRUELAT2),float(ds.STAND_LON),float(ds.DX),float(ds.DY))
        run_start=datetime.strptime(ds['Times'][0].tobytes().decode(),'%Y-%m-%d_%H:%M:%S')
    reports=read_obs_domain(obsfile)
    data=prepare_reports(reports,projection,run_start,options,dt_s,nx,ny)
    digest=hashlib.sha256(obsfile.read_bytes())
    digest.update((root/f'wrfinput_{domain}').read_bytes())
    digest.update(json.dumps(dict(options=options,dt_s=dt_s,nx=nx,ny=ny,start=run_start.isoformat(),
                             projection={k:getattr(projection,k) for k in
                              ('code','lat1','lon1','truelat1','truelat2','stdlon','dx','dy')}),
                             sort_keys=True,default=float).encode())
    if options['obs_ipf_init']:
        print(f'{domain}: observation nudging: {len(reports)} reports, '
              f'rolling frame {data["max_reports"]} reports/{data["max_observation_levels"]} levels; {obsfile.name}')
    return ObservationInputs(**{k:jnp.asarray(data[k]) for k in
            ('times','arrivals','deactivated','rio','rjo','rko','variants','variant_steps','lengths')},
            options=tuple(sorted(options.items())),max_reports=data['max_reports'],fingerprint=digest.hexdigest())
