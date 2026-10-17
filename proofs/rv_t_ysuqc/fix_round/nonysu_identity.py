"""Compare only the FIX commit's operational forcing change on the same final globals.

No model execution: source-location-normalized full forcing jaxpr identities.
The pre-FIX function is loaded in memory from a164805bc3; no checkout is mutated.
"""
from pathlib import Path
import ast, dataclasses, hashlib, json, re, subprocess
import jax
from gpuwrf.runtime import operational_mode as om
from gpuwrf.runtime.operational_state import initial_operational_carry
from test_v013_operational_smoke import _grid, _base_state, _namelist

OUT=Path('<USER_HOME>/wrf_gpu2_lanes/rv-t-ysuqc/fix_round')
source=subprocess.check_output(['git','show','a164805bc3:src/gpuwrf/runtime/operational_mode.py'],text=True)
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_physics_step_forcing')
scope=dict(om.__dict__)
exec(compile(ast.Module(body=[node],type_ignores=[]),'<pre-FIX forcing>','exec'),scope)
old=scope['_physics_step_forcing'];new=om._physics_step_forcing
grid=_grid(nz=12,ny=3,nx=3)
configs={
    'mynn5_ra1_1':dict(bl_pbl_physics=5,sf_sfclay_physics=5,ra_lw_physics=1,ra_sw_physics=1),
    'shinhong11_ra1_1':dict(bl_pbl_physics=11,sf_sfclay_physics=1,ra_lw_physics=1,ra_sw_physics=1),
    'acm2_7_ra1_1':dict(bl_pbl_physics=7,sf_sfclay_physics=1,ra_lw_physics=1,ra_sw_physics=1),
    'boulac8_ra4_4':dict(bl_pbl_physics=8,sf_sfclay_physics=1,ra_lw_physics=4,ra_sw_physics=4),
    'mynn5_ra0_0':dict(bl_pbl_physics=5,sf_sfclay_physics=5,ra_lw_physics=0,ra_sw_physics=0),
}
result={}
for name,cfg in configs.items():
    for rk in (0,1):
        for rad in (True,False):
            ns=_namelist(grid,mp_physics=0,cu_physics=0,sf_surface_physics=0,use_noahmp=False,rad_rk_tendf=rk,**cfg)
            carry=initial_operational_carry(_base_state(grid))
            hashes={}
            for label,fn in [('old',old),('new',new)]:
                def call(c):
                    f=fn(c,ns,0.,run_radiation=rad,first_timestep=True)
                    leaves=jax.tree_util.tree_leaves((f.state,f.carry))
                    for fld in dataclasses.fields(f.dry_tendencies):
                        leaves+=jax.tree_util.tree_leaves(getattr(f.dry_tendencies,fld.name))
                    return [x for x in leaves if hasattr(x,'shape')]
                text=str(jax.make_jaxpr(call)(carry))
                text=re.sub(r'/[^ \]\)\"]*\.py:\d+(:\d+)?','',text)
                text=re.sub(r'name_and_src_info=[^ ,\]]*','',text)
                hashes[label]=hashlib.sha256(text.encode()).hexdigest()
            hashes['same']=hashes['old']==hashes['new']
            result[f'{name}_rk{rk}_rad{int(rad)}']=hashes
            assert hashes['same'],(name,rk,rad,hashes)
report=dict(candidate='5e6a43634c',pre_fix='a164805bc3',scope='isolated forcing function FIX delta, same final module globals; CPU trace only',configs=result,all_same=True)
(OUT/'nonysu_identity.json').write_text(json.dumps(report,indent=2)+'\n')
print('20/20 non-YSU forcing programs identical')
