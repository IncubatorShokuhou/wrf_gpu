"""Actual land table bindings separate compiled constants; runtime maps stay dynamic."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.runtime import aot_cheap_key as ck,aot_executable as ax
from gpuwrf.runtime.aot_baked_tables import baked_table_hash
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.contracts.grid import GridSpec
from gpuwrf.io.noahmp_options import NoahMPOptions,read_bats_parameters
from gpuwrf.physics.noahmp.carbon import CarbonStatic


def namelist(**kwargs):
    return OperationalNamelist(GridSpec.canary_3km_template(),None,None,18.,
        mp_physics=0,ra_sw_physics=0,ra_lw_physics=0,sf_sfclay_physics=0,cu_physics=0,**kwargs)


def test_every_new_land_external_reader_has_an_explicit_key_disposition():
    import ast
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/'src/gpuwrf/io'
    dispositions={'noahmp_options.py':'bound-value provider/static auxiliary',
        'noahmp_soil.py':'NoahMPStatic content holder',
        'noahmp_hydrology.py':'NoahMPStatic content holder',
        'noahmp_carbon.py':'traced child/closed-child content identity',
        'noahmp_crop.py':'traced child plus energy holder/closed-child identity',
        'noahmp_irrigation.py':'traced child/closed-child content identity',
        'noahmp_mmf.py':'inherited main MMF static holder/full namelist identity'}
    readers={path.name for path in root.glob('noahmp_*.py') for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
        and node.func.attr in ('read_text','read_bytes','load','fromfile','loadtxt','genfromtxt')}
    assert readers==set(dispositions)


@pytest.mark.parametrize('field',('bats','rsurf_snow','fsatmx','timean','drain_layer_opt'))
def test_actual_bound_mptable_controls_change_provider_and_full_key(field):
    opts=NoahMPOptions(opt_alb=1,bats=read_bats_parameters(),opt_rsf=4,rsurf_snow=20.,
        opt_run=2,fsatmx=.38,timean=10.)
    if field=='drain_layer_opt':opts=replace(opts,opt_run=3,opt_tdrn=1,drain_layer_opt=4)
    value=getattr(opts,field)
    changed=tuple([value[0]+.125,*value[1:]]) if field=='bats' else value+1
    if field=='fsatmx':changed=.5
    values=[]
    for selected in (opts,replace(opts,**{field:changed})):
        nl=namelist(noahmp_options=selected)
        @jax.jit
        def kernel(x,nl):
            v=getattr(nl.noahmp_options,field)
            return x+(v[0] if field=='bats' else v)
        x=jnp.ones(1,jnp.float32)
        values.append((baked_table_hash(kernel,nl),ck.cheap_key(kernel,(x,nl),{},nl),
            ax.hlo_sha256_from_lowered(kernel.lower(x,nl)),np.asarray(kernel(x,nl))))
    assert all(values[0][i]!=values[1][i] for i in range(3))
    assert not np.array_equal(values[0][3],values[1][3])


def test_crop_table_child_is_dynamic_or_content_keyed_when_closed_over():
    # A real CarbonStatic child is the binding destination of crop/carbon
    # MPTABLE arrays. Ordinary calls pass it as an operand; fused calls bake it.
    a=namelist(noahmp_carbon_static=CarbonStatic(jnp.asarray([8.],jnp.float32),None))
    b=replace(a,noahmp_carbon_static=CarbonStatic(jnp.asarray([9.],jnp.float32),None))
    @jax.jit
    def direct(x,nl):return x+nl.noahmp_carbon_static.parameters
    x=jnp.ones(1,jnp.float32)
    assert ck.cheap_key(direct,(x,a),{},a)==ck.cheap_key(direct,(x,b),{},b)
    def factory(nl):
        @jax.jit
        def kernel(x):return x+nl.noahmp_carbon_static.parameters
        return kernel
    records=[]
    for nl in (a,b):
        fn=factory(nl)
        records.append((ck.cheap_key(fn,(x,),{},nl),ax.hlo_sha256_from_lowered(fn.lower(x)),np.asarray(fn(x))))
    assert records[0][0]!=records[1][0] and records[0][1]!=records[1][1]
    assert not np.array_equal(records[0][2],records[1][2])


@pytest.mark.parametrize('group',('runoff','tile','tile_fraction','soil'))
def test_selected_hydrology_tables_are_in_actual_static_holder(group):
    from test_noahmp_coupler import _build
    from gpuwrf.io.noahmp_hydrology import SelectedHydrologyParameters
    from gpuwrf.physics.noahmp.tables import NoahMPParameters
    from gpuwrf.physics.noahmp.runoff_options import RunoffParameters,HydrologyParameters
    from gpuwrf.physics.noahmp.tile_drain import TileParameters
    _,_,static,_,_=_build()
    values=jnp.ones((4,1,5),jnp.float32)
    hydro=HydrologyParameters(RunoffParameters(*(values for _ in RunoffParameters._fields)),
        TileParameters(*(jnp.ones((1,5),jnp.float32) for _ in TileParameters._fields)),
        jnp.ones((1,5),jnp.float32),jnp.ones((1,5),jnp.float32)*3000.)
    params=SelectedHydrologyParameters(*(getattr(static.parameters,k) for k in NoahMPParameters._fields),
        values,values,jnp.ones((4,1,5),jnp.int32),hydro)
    def factory(nl):
        @jax.jit
        def kernel(x):
            p=nl.noahmp_static.parameters
            selected=(p.hydrology.runoff.bvic if group=='runoff' else p.hydrology.tile.td_dc if group=='tile'
                      else p.hydrology.tile_fraction if group=='tile_fraction' else p.kdt)
            return x+jnp.sum(selected)
        return kernel
    if group=='runoff':changed=params._replace(hydrology=hydro._replace(runoff=hydro.runoff._replace(bvic=values+1.)))
    elif group=='tile':changed=params._replace(hydrology=hydro._replace(tile=hydro.tile._replace(td_dc=hydro.tile.td_dc+1.)))
    elif group=='tile_fraction':changed=params._replace(hydrology=hydro._replace(tile_fraction=hydro.tile_fraction+.25))
    else:changed=params._replace(kdt=values+1.)
    x=jnp.ones(1,jnp.float32);records=[]
    for p in (params,changed):
        nl=namelist(noahmp_static=static.replace(parameters=p));fn=factory(nl)
        records.append((ck.cheap_key(fn,(x,),{},nl),ax.hlo_sha256_from_lowered(fn.lower(x)),np.asarray(fn(x))))
    assert records[0][0]!=records[1][0] and records[0][1]!=records[1][1]
    assert not np.array_equal(records[0][2],records[1][2])
