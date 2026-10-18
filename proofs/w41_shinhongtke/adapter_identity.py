"""OFF-identity of the bl=11 adapter for shinhong_tke_diag=0 (E151 style, CPU, seconds).

Loads main's coupling/scan_adapters.py (git show <ref>:path) as a second module object next
to the lane's, traces both shinhong_pbl_adapter calls on the same small real-structured State
(tests/test_rrtm_lw_operational_wiring fixture), strips source locations, and compares the
jaxpr text. Expected: diag=0 identical (no program change), diag=1 differs only by the el_pbl
output. usage: python adapter_identity.py <repo> <git_ref> <out.json>
"""
import importlib.util, json, re, subprocess, sys, tempfile
from pathlib import Path

repo, ref, out = Path(sys.argv[1]).resolve(), sys.argv[2], Path(sys.argv[3])
sys.path.insert(0, str(repo / 'src'))
import jax, jax.numpy as jnp
assert jax.devices()[0].platform == 'cpu'
from gpuwrf.coupling import scan_adapters as new
old_src = subprocess.run(['git', 'show', f'{ref}:src/gpuwrf/coupling/scan_adapters.py'], cwd=repo,
                         capture_output=True, text=True, check=True).stdout
tmp = Path(tempfile.mkdtemp()) / 'scan_adapters_old.py'; tmp.write_text(old_src)
spec = importlib.util.spec_from_file_location('gpuwrf.coupling.scan_adapters_old', tmp)
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
fspec = importlib.util.spec_from_file_location('fx', repo / 'tests/test_rrtm_lw_operational_wiring.py')
f = importlib.util.module_from_spec(fspec); fspec.loader.exec_module(f)
grid = f._grid(nz=8, ny=2, nx=2)
state = f._state(grid).ensure_conditional_leaves(mp_physics=8)
state = state.replace(qke=jnp.full_like(state.qke, jnp.float32(.005)))
strip = lambda text: re.sub(r'\S+\.py:\d+(:\d+)?', 'LOC', re.sub(r'name_and_src_info=\S+', '', text))
res = {}
for diag in (0, 1):
    texts = {}
    for tag, mod in (('main', old), ('lane', new)):
        fn = lambda s, m=mod: jax.tree.leaves(m.shinhong_pbl_adapter(s, 18., grid, shinhong_tke_diag=diag))
        texts[tag] = strip(str(jax.make_jaxpr(fn)(state)))
    res[f'diag{diag}'] = dict(identical=texts['main'] == texts['lane'],
                              main_len=len(texts['main']), lane_len=len(texts['lane']))
res['ref'] = subprocess.run(['git', 'rev-parse', ref], cwd=repo, capture_output=True, text=True).stdout.strip()
res['lane_head'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, capture_output=True, text=True).stdout.strip()
out.write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res))
assert res['diag0']['identical'] and not res['diag1']['identical']
