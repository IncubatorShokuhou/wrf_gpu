"""Natural-soil option carries must hold on WRF's independent glacier tiles."""
from types import SimpleNamespace
import jax.numpy as jnp
import numpy as np
import pytest
from test_s2_glacier_wiring import setup
from gpuwrf.contracts.noahmp_state import NoahMPFluxes
from gpuwrf.coupling.noahmp_option_tiles import option_glacier_step


@pytest.mark.parametrize('gla',(1,2))
def test_options_and_all_four_carries_follow_natural_partition(gla):
    land,forcing,static,_,_=setup(gla)
    original=jnp.asarray([[2.,9.]],jnp.float32)
    seen=[]
    def natural(l,f,s,dt,*,energy_params,rad_params,history,options,irrigation,carbon,groundwater,tile_drain):
        assert l.tg.shape==(1,1) and options.opt_btr==3
        for value in (irrigation[0],irrigation[1],carbon[0],carbon[1],groundwater,tile_drain):
            np.testing.assert_array_equal(value,[[2.]])
        seen.append(True);zero=jnp.zeros_like(l.tg)
        flux=NoahMPFluxes(zero,zero,zero,zero,l.t_skin,l.qsfc,l.znt,l.emiss,l.albedo,zero,
            history={'BGAP':jnp.ones_like(l.tg)*.25,'GPP':jnp.ones_like(l.tg)*.125})
        return l,flux,irrigation[1]+1.,carbon[1]+2.,groundwater+3.,tile_drain+4.
    _,flux,irr,carbon,groundwater,tile=option_glacier_step(natural,land,forcing,static,18.,
        energy_params=None,rad_params=None,history=True,options=SimpleNamespace(opt_btr=3),
        irrigation=(original,original),carbon=(original,original),groundwater=original,tile_drain=original)
    assert seen==[True]
    np.testing.assert_array_equal(flux.history['BGAP'],[[.25,0.]])
    np.testing.assert_array_equal(flux.history['GPP'],[[.125,0.]])
    for value,increment in zip((irr,carbon,groundwater,tile),(1,2,3,4),strict=True):
        np.testing.assert_array_equal(value,[[2.+increment,9.]])
