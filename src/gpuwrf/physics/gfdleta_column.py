"""Explicit single-column boundary for the qualified WRF GFDL RADTN port.

The source transcription is ordinary JAX, with no runtime parser/Fortran
forecast callback. Its only host provider is initialized before stepping.
"""
from functools import lru_cache
import hashlib
import jax.numpy as jnp
import numpy as np
from gpuwrf.physics import ra_gfdleta as optics
from gpuwrf.physics.ra_gfdleta_support import zenith_dynamic,solar_distance
from gpuwrf.physics.gfdleta_co2 import co2_tables

ARGUMENTS=('dt','t','q','qcw','qice','pflip','p8wflip','xland','tsk2d','glat','glon',
 'cutop','cubot','alb','cuppt','acfrcv','ncfrcv','acfrst','ncfrst','vegfrc','sno',
 'glw','gsw','rswin','rswinc','idat','ihrst','xtime','julian','nrads','nradl',
 'nphs','ntsd','tends','tendl','cldfra','rswtoa','rlwtoa','czmean','cfracl',
 'cfracm','cfrach','ids','ide','jds','jde','kds','kde','ims','ime','jms','jme',
 'kms','kme','its','ite','jts','jte','kts','kte')
OUTPUTS=('acfrcv','ncfrcv','acfrst','ncfrst','glw','gsw','rswin','rswinc','tends',
 'tendl','cldfra','rswtoa','rlwtoa','czmean','cfracl','cfracm','cfrach')


@lru_cache(maxsize=8)
def _solver(full_bytes,half_bytes,top_bytes,provider_identity):
    from gpuwrf.physics._gfdleta_radtn import reference_factory
    full=np.frombuffer(full_bytes,np.float32);half=np.frombuffer(half_bytes,np.float32)
    co2,domain=co2_tables(full,half,np.frombuffer(top_bytes,np.float32)[0])
    solver=reference_factory(len(half),co2,domain,optics.cloud_overlap,
        optics.ferrier_saturation_pressure,optics.ozone_column,zenith_dynamic)
    captured=dict(zip(solver.__code__.co_freevars,(c.cell_contents for c in solver.__closure__)))
    solver.gpuwrf_baked_tables={'spectral':captured['_TABLES'],'cloud':optics._host_tables()}
    return solver


def prepare_solver(eta_interfaces,eta_midpoints,top_pressure_pa):
    """Resolve host tables before scan construction; cached closure holds NumPy only."""
    from gpuwrf.physics import _gfdleta_radtn as model
    full=np.asarray(eta_interfaces,np.float32);half=np.asarray(eta_midpoints,np.float32)
    top=np.asarray(top_pressure_pa,np.float32)
    co2,domain=co2_tables(full,half,top)
    identity=hashlib.sha256()
    for group in (model._host_tables(),optics._host_tables(),co2,domain):
        for key,value in sorted(group.items()):
            array=np.asarray(value)
            identity.update(key.encode()+str((array.shape,array.dtype.str)).encode()+array.tobytes())
    return _solver(full.tobytes(),half.tobytes(),top.tobytes(),identity.hexdigest())


def native_column(solver,operands,clock,previous,*,dt,stepra,nphs=1):
    """Call the literal native ABI with explicit physical arrays and saved state.

    Profiles run bottom-to-top at this boundary. Coordinates are radians,
    vegetation is percent, snow is native WRF kg/m2 and CUPPT is metres.
    clock=(initial month/day/year, NINT(initial GMT), forecast minutes, JULIAN).
    """
    n=operands['t'].shape[0]
    f=jnp.float32;i=jnp.int32
    profile=lambda a:a[::-1][None,:,None]
    surface=lambda a:jnp.asarray(a).reshape(1,1)
    zero_surface=jnp.zeros((1,1),f)
    values={key:i(1) for key in ARGUMENTS}
    for key in ('ide','jde'):values[key]=i(2)
    for key in ('kde','kme'):values[key]=i(n+1)
    values['kte']=i(n)
    values.update(dt=f(dt),t=profile(operands['t']),pflip=profile(operands['p']),
        p8wflip=profile(operands['p8w']),q=profile(jnp.maximum(0.,operands['qv']/(1.+operands['qv']))),
        qcw=profile(jnp.maximum(operands['qc'],0.)),
        qice=profile(jnp.maximum(operands['qs'],0.)+operands['qi']))
    for native,key in {'xland':'xland','tsk2d':'tsk','glat':'latitude','glon':'longitude',
                        'alb':'albedo','vegfrc':'vegetation','sno':'snow'}.items():
        values[native]=surface(f(operands[key]))
    values['cutop']=surface(f(n+1)-previous['top'])
    values['cubot']=surface(f(n+1)-previous['bottom'])
    values['cuppt']=surface(previous['cuppt'])
    for key in ('acfrcv','ncfrcv','acfrst','ncfrst'):
        if key in ('ncfrcv','ncfrst') and jnp.asarray(previous[key]).dtype!=jnp.int32:
            raise TypeError('GFDL cloud counter must be native INTEGER32: '+key)
        values[key]=surface(previous[key])
    for key in OUTPUTS:
        if key not in ('acfrcv','ncfrcv','acfrst','ncfrst'):
            values[key]=jnp.zeros((1,n,1),f) if key in ('tends','tendl') else (
                jnp.zeros((1,n+1,1),f) if key=='cldfra' else zero_surface)
    date,start_hour,minutes,julian=clock
    hour=jnp.fmod(start_hour+jnp.floor(minutes/60.+.5).astype(i),i(24))
    distance=solar_distance(hour,date[1],date[0],date[2])
    values.update(idat=date,ihrst=start_hour,xtime=f(minutes),julian=f(julian),
        nrads=i(stepra),nradl=i(stepra),nphs=i(nphs),ntsd=i(1)+jnp.floor(minutes*60./f(dt)+.5).astype(i))
    result=solver({'r1':distance},*(values[key] for key in ARGUMENTS))
    return {key:result[ARGUMENTS.index(key)] for key in OUTPUTS}
