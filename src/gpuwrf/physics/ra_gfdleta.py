"""GFDL radiation source components; operational ra=99 admission remains closed.

Table lookups retain WRF's Fortran flat indexing and REAL arithmetic. These
are column reference components; a native GPU solver must embed its vertical
recurrences inside a column kernel rather than domain-vmap the reference.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from jax import lax


@lru_cache(maxsize=1)
def _host_tables():
    # Cache host arrays only: never memoize values produced under a JAX trace.
    with np.load(Path(__file__).with_name('gfdleta_tables_v1.npz'),allow_pickle=False) as f:
        tables={key:f[key].copy() for key in f.files}
    for value in tables.values():
        if value.dtype!=np.float32 or not np.isfinite(value).all():
            raise ValueError('Invalid native GFDL table asset')
        value.setflags(write=False)
    return tables


def _table(name,dtype):
    return jnp.asarray(_host_tables()[name],dtype)


def preprocess_column(temperature,pressure,p8w,qv,qc,qs,htop,hbot,*,qi=None):
    """ETARA caller mapping; mass arrays include their unused padded top slot.

    GFDL ignores rain and graupel. Snow is clipped before adding optional cloud
    ice, which the original caller does not clip. Mixing ratio becomes specific
    humidity here, before RADTN's EPSQ floor.
    """
    nz=temperature.shape[-1]-1
    reverse=lambda a:a[..., :-1][..., ::-1]
    vapor=reverse(qv)
    ice=jnp.maximum(reverse(qs),0.)
    if qi is not None:
        ice=ice+reverse(qi)
    return dict(t=reverse(temperature),p=reverse(pressure),p8w=p8w[..., ::-1],
                q=jnp.maximum(0.,vapor/(1.+vapor)),qc=jnp.maximum(reverse(qc),0.),
                qi=ice,cutop=nz+1.-htop,cubot=nz+1.-hbot)


def ferrier_saturation_pressure(temperature):
    """Active GFDL Ferrier FPVS lookup, kPa, including native range handling.

    This is not the GFS mixed-phase table or the inactive FPVS_new polynomial.
    """
    c1,c2=_table('ferrier_coefficients',temperature.dtype)
    table=_table('ferrier_tbpvs',temperature.dtype)
    xj=jnp.minimum(jnp.maximum(c1+c2*temperature,1.),7501.)
    jx=jnp.minimum(xj,7500.).astype(jnp.int32)
    # WRF converts INTEGER JX to REAL for this subtraction. Make that explicit
    # also for literal evaluation (NumPy otherwise promotes REAL4-int32 to f64).
    return table[jx-1]+(xj-jx.astype(temperature.dtype))*(table[jx]-table[jx-1])


def grid_cloud_fraction(temperature,pressure,specific_humidity,water,ice,cutop,cubot,*,ep2):
    """RADTN Ferrier Gaussian cloud fraction for its active LVL=0 column.

    q is already specific humidity. Returned fraction is zero without native
    condensate, preserving the source skip branch rather than inventing cloud.
    cutop/cubot are native radiation-order one-based convection bounds.
    """
    dtype=temperature.dtype
    q=jnp.maximum(specific_humidity,1.e-12)
    vapor=q/(1.-q)
    condensate=water+ice
    esat=1000.*ferrier_saturation_pressure(temperature)
    qsat=jnp.asarray(ep2,dtype)*esat/(pressure-esat)
    total_humidity=(vapor+condensate)/qsat
    n=temperature.shape[-1]
    nint=lambda x:jnp.where(x>=0.,jnp.floor(x+.5),jnp.ceil(x-.5)).astype(jnp.int32)
    top=jnp.minimum(n,nint(cutop));bottom=jnp.minimum(n,nint(cubot))
    levels=jnp.arange(n,dtype=jnp.int32)+1
    deviation=jnp.where((levels>=top) & (levels<=bottom),jnp.asarray(.04,dtype),jnp.asarray(.01,dtype))
    arg=(total_humidity-1.)/deviation
    index=jnp.minimum(310,jnp.maximum((jnp.abs(arg)/.01+.5).astype(jnp.int32),1))
    area=_table('axsd',dtype)[index-1]
    negative=.5-area
    negative=jnp.where(negative<.01,0.,negative)
    fraction=jnp.where(arg>=3.1,1.,jnp.where(arg<=-3.1,0.,
                jnp.where((arg<=.005) & (arg>=-.005),.5,jnp.where(arg>.005,.5+area,negative))))
    return jnp.where(condensate<=1.e-12,0.,fraction)


def ozone_column(pressure,latitude,rsin1,rcos1,rcos2):
    """OZON2D: seasonal/latitude mean followed by logarithmic pressure lookup.

    Pressure is the actual POZN caller operand; RADTN's below-top convention
    must be preserved by its caller rather than replaced with mass pressure.
    """
    dtype=pressure.dtype
    th2=.2*latitude
    row=(19.001-th2).astype(jnp.int32)
    fraction=(jnp.int32(19)-row).astype(dtype)-th2
    avg,a1,b1,b2=(_table(k,dtype) for k in ('xduo3n','xdo3n2','xdo3n3','xdo3n4'))
    lower=avg[row-1]+rsin1*a1[row-1]+rcos1*b1[row-1]+rcos2*b2[row-1]
    upper=avg[row]+rsin1*a1[row]+rcos1*b1[row]+rcos2*b2[row]
    profile=1.e-4*(lower+fraction*(upper-lower))
    p=_table('prgfdl',dtype)
    index=jnp.clip(jnp.searchsorted(p,pressure,side='right'),1,80)
    lo=jnp.log(p[index-1]);hi=jnp.log(p[index])
    value=profile[index]+(jnp.log(pressure)-hi)/(lo-hi)*(profile[index-1]-profile[index])
    return jnp.where(pressure<p[0],profile[0],jnp.where(pressure>p[-1],profile[-1],value))


def emissivity_e2(amount, temperature_index, delta_temperature):
    """Shared E290/E2SPEC lookup expression; caller owns special averaging.

    temperature_index is WRF's one-based FXOE2/FXOSP, not temperature itself.
    No clipping: invalid physical table operands must remain visible to gates.
    """
    tmp=jnp.log10(amount)+16.
    fyo=jnp.trunc(tmp*10.)
    du=tmp-.1*fyo
    index=(28.*fyo+temperature_index).astype(jnp.int32)-1
    t1,t2,t4=(_table(key,amount.dtype).T.reshape(-1) for key in ('table1','table2','table3'))
    return t1[index]+du*t2[index]+delta_temperature*t4[index]


def emissivity_e3(temperature,amount):
    """E3V88 bilinear lookup, including native table-start index offsets."""
    fxo=jnp.trunc(temperature*.1)
    tmp=jnp.log10(amount)+16.
    dt=temperature-10.*fxo
    fyo=jnp.trunc(tmp*10.)
    du=tmp-.1*fyo
    index=(fxo+fyo*28.).astype(jnp.int32)
    ww1=10.-dt;ww2=.1-du
    table=_table('em3',temperature.dtype).T.reshape(-1)
    return (ww1*ww2*table[index-10]+ww2*dt*table[index-9]+
            ww1*du*table[index+18]+dt*du*table[index+19])


def cloud_overlap(amount,bottom,top,number):
    """Single-column CLO89 random cloud overlap, native one-based operands.

    Slot zero is the native surface pseudo-cloud. Actual clouds begin at slot
    one. Return an interface-by-interface transmission matrix.
    """
    levels=jnp.arange(amount.shape[0],dtype=jnp.int32)+1
    rows=levels[:,None];columns=levels[None,:]
    initial=jnp.ones((amount.shape[0],amount.shape[0]),amount.dtype)

    def cloud(n,result):
        k1=top[n]+1;k2=bottom[n]
        # These are CLO89's three non-overlapping column ranges. In the gap
        # K2+1..K1-1 the cloud factor is one, exactly as native initialization.
        crosses=((columns>=jnp.maximum(k1,k2+1)) & (rows<=k2) |
                 (columns<=jnp.minimum(k1-1,k2)) & (rows>=k1) |
                 (columns>=k1) & (columns<=k2))
        factor=jnp.where(crosses,1.-amount[n],jnp.ones_like(result))
        return result*factor

    return lax.fori_loop(1,number+1,cloud,initial)
