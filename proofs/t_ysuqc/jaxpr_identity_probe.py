import sys, re, json, hashlib
root = sys.argv[1]; out = sys.argv[2]
sys.path.insert(0, root + '/src'); sys.path.insert(0, root + '/tests')
import jax, jax.numpy as jnp
from gpuwrf.runtime import operational_mode as om
from gpuwrf.runtime.operational_state import initial_operational_carry
from test_v013_operational_smoke import _grid, _base_state, _namelist
assert om.__file__.startswith(root), om.__file__
grid = _grid(nz=12, ny=3, nx=3)
configs = {
  'mynn5_ra1_1': dict(bl_pbl_physics=5, sf_sfclay_physics=5, ra_lw_physics=1, ra_sw_physics=1),
  'shinhong11_ra1_1': dict(bl_pbl_physics=11, sf_sfclay_physics=1, ra_lw_physics=1, ra_sw_physics=1),
  'acm2_7_ra1_1': dict(bl_pbl_physics=7, sf_sfclay_physics=1, ra_lw_physics=1, ra_sw_physics=1),
  'boulac8_ra4_4': dict(bl_pbl_physics=8, sf_sfclay_physics=1, ra_lw_physics=4, ra_sw_physics=4),
  'mynn5_ra0_0': dict(bl_pbl_physics=5, sf_sfclay_physics=5, ra_lw_physics=0, ra_sw_physics=0),
  'ysu1_ra1_1': dict(bl_pbl_physics=1, sf_sfclay_physics=1, ra_lw_physics=1, ra_sw_physics=1),
  'ysu1_ra0_0': dict(bl_pbl_physics=1, sf_sfclay_physics=1, ra_lw_physics=0, ra_sw_physics=0),
}
res = {}
for name, cfg in configs.items():
    for rk in (0, 1):
        for rad in (True, False):
            ns = _namelist(grid, mp_physics=0, cu_physics=0, sf_surface_physics=0, use_noahmp=False, rad_rk_tendf=rk, **cfg)
            carry = initial_operational_carry(_base_state(grid))
            try:
                def fn(c):
                    import dataclasses
                    f = om._physics_step_forcing(c, ns, 0., run_radiation=rad, first_timestep=True)
                    leaves = jax.tree_util.tree_leaves((f.state, f.carry))
                    for fld in dataclasses.fields(f.dry_tendencies):
                        leaves += jax.tree_util.tree_leaves(getattr(f.dry_tendencies, fld.name))
                    return [x for x in leaves if hasattr(x, 'shape')]
                jx = jax.make_jaxpr(fn)(carry)
                txt = re.sub(r'/[^ \]\)"]*\.py:\d+(:\d+)?', '', str(jx))
                txt = re.sub(r'name_and_src_info=[^ ,\]]*', '', txt)
                res[f'{name}_rk{rk}_rad{int(rad)}'] = hashlib.sha256(txt.encode()).hexdigest()[:16]
            except Exception as e:
                res[f'{name}_rk{rk}_rad{int(rad)}'] = 'ERR ' + type(e).__name__ + ': ' + str(e)[:300]
json.dump(res, open(out, 'w'), indent=1); print(json.dumps(res, indent=1))
