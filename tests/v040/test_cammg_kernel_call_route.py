"""mp11 native column must use the stock-Triton + triton_kernel_call route.

XLA's bundled Triton miscompiles the CAM-MG column on sm_120 (zero
RAINNCV/SNOWNCV; lane t-cammgws CG02-CG19). These CPU gates pin (a) that the
column selects KernelCallCompilerParams, (b) that the registered GPU rule emits
`triton_kernel_call` with PTX produced by STOCK Triton (not XLA's bundled one), and
(c) that the route refuses a lowering without a visible CUDA device instead of
generating PTX for a guessed architecture. Deleting the route fails (a) and (b).
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from gpuwrf.kernels import _triton_kernel_call as route


def _copy_kernel(x_ref,o_ref):
    o_ref[...]=x_ref[...]*2.0


def _lowered(params,local_devices,monkeypatch):
    monkeypatch.setattr(jax,'local_devices',local_devices)
    f=lambda x:pl.pallas_call(_copy_kernel,out_shape=jax.ShapeDtypeStruct((16,),jnp.float32),
                              compiler_params=params,name='route_probe')(x)
    return jax.jit(f).trace(jax.ShapeDtypeStruct((16,),jnp.float32)).lower(lowering_platforms=('cuda',)).as_text()


_PROBE = r"""
import sys
from types import SimpleNamespace
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from gpuwrf.kernels import _triton_kernel_call as route
assert jax.devices()[0].platform=='cpu'
try:
    route._stock_triton()
except RuntimeError:
    print('NOTRITON');sys.exit(0)
jax.local_devices=lambda *a,**k:[SimpleNamespace(compute_capability='12.0',platform='gpu')]
def kernel(x_ref,o_ref):o_ref[...]=x_ref[...]*2.0
for label,params in (('route',route.KernelCallCompilerParams(num_warps=4)),('default',route.triton_core.CompilerParams(num_warps=4))):
    f=lambda x:pl.pallas_call(kernel,out_shape=jax.ShapeDtypeStruct((16,),jnp.float32),compiler_params=params,name='route_probe')(x)
    text=jax.jit(f).trace(jax.ShapeDtypeStruct((16,),jnp.float32)).lower(lowering_platforms=('cuda',)).as_text()
    print(label,'triton_kernel_call' in text,'__gpu$xla.gpu.triton' in text)
"""


def test_registered_rule_emits_triton_kernel_call():
    # Separate process with NO visible CUDA device (E50): stock Triton compiles on
    # the host; only the architecture query is faked at lowering.
    import os,subprocess,sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    env={**os.environ,'JAX_PLATFORMS':'cpu','CUDA_VISIBLE_DEVICES':'','GPUWRF_JAX_CACHE':'0','PYTHONPATH':str(root/'src')}
    result=subprocess.run([sys.executable,'-c',_PROBE],cwd=root,env=env,capture_output=True,text=True,timeout=300)
    assert result.returncode==0,result.stdout+result.stderr[-3000:]
    if 'NOTRITON' in result.stdout:pytest.skip('stock Triton 3.6 not installed')
    lines=dict(line.split(' ',1) for line in result.stdout.split('\n') if line.startswith(('route ','default ')))
    assert lines['route']=='True False',result.stdout
    assert lines['default']=='False True',result.stdout


def test_route_refuses_without_cuda_device(monkeypatch):
    def none(*a,**k):raise RuntimeError('no gpu')
    with pytest.raises(Exception,match='visible CUDA device'):
        _lowered(route.KernelCallCompilerParams(num_warps=4),none,monkeypatch)


def test_cammg_column_selects_kernel_call_route(monkeypatch):
    import gpuwrf.backends.pallas as facade
    from gpuwrf.kernels import phys_cammg
    # The kernel imports the portability facade: hold it at its CUDA dispatch (the stock
    # pallas_call object, as on a CUDA host); on CPU it would interpret and drop compiler_params.
    monkeypatch.setattr(facade,'cuda_native',lambda:True)
    assert phys_cammg.pl is facade
    with np.load(__import__('pathlib').Path(__file__).resolve().parents[2]/'proofs/s2_cammg/fixtures/public.npz') as z:
        fields={k.split('_input_',1)[1]:jnp.asarray(z[k]).reshape((1,) if k.split('_input_',1)[1] in phys_cammg.SURFACE_FIELDS else (1,25))
                for k in z.files if k.startswith('c01_s1_input_')}
    with jax.enable_x64(False):
        program=jax.make_jaxpr(lambda f:phys_cammg.full_column(f,18.,jnp.int32(1)))(fields).jaxpr
    params=[e.params['compiler_params'] for e in program.eqns if e.primitive.name=='pallas_call']
    assert len(params)==1 and type(params[0]) is route.KernelCallCompilerParams and params[0].num_warps==4


def test_stock_compiler_identity_reaches_mp11_aot_key(monkeypatch):
    # t-aotkey rule: the PTX generator baked into the mp11 executable must key it.
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.runtime.aot_baked_tables import baked_table_hash
    nl=lambda mp:OperationalNamelist(GridSpec.canary_3km_template(),None,None,18.,mp_physics=mp,ra_sw_physics=0,
                                     ra_lw_physics=0,sf_sfclay_physics=0,cu_physics=0)
    fn=lambda x,nl:x
    before,other=baked_table_hash(fn,nl(11)),baked_table_hash(fn,nl(8))
    monkeypatch.setattr(route,'stock_compiler_identity',lambda:('stock-triton','9.9.9','elsewhere',1,2))
    assert baked_table_hash(fn,nl(11))!=before
    assert baked_table_hash(fn,nl(8))==other


def _twin_trees(tmp_path):
    # Same relative names, sizes and timestamps; one byte differs (critic rv-t-cammgws item 1).
    import os
    a,b=tmp_path/'a',tmp_path/'b'
    for root,byte in ((a,b'\x00'),(b,b'\x01')):
        (root/'_C').mkdir(parents=True)
        (root/'_C'/'libtriton.so').write_bytes(b'ELF'+byte*64)
        (root/'compiler.py').write_text('stage = 1\n')
        for path in ((root/'_C'/'libtriton.so'),(root/'compiler.py')):os.utime(path,ns=(10**18,10**18))
    return a,b


def test_tree_digest_is_content_addressed(tmp_path):
    a,b=_twin_trees(tmp_path)
    assert route.tree_digest(a)!=route.tree_digest(b)
    (b/'_C'/'libtriton.so').write_bytes((a/'_C'/'libtriton.so').read_bytes())
    assert route.tree_digest(a)==route.tree_digest(b)


def test_same_metadata_compiler_swap_changes_mp11_key_only(tmp_path,monkeypatch):
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.runtime.aot_baked_tables import baked_table_hash
    nl=lambda mp:OperationalNamelist(GridSpec.canary_3km_template(),None,None,18.,mp_physics=mp,ra_sw_physics=0,
                                     ra_lw_physics=0,sf_sfclay_physics=0,cu_physics=0)
    fn=lambda x,nl:x
    a,b=_twin_trees(tmp_path)
    keys={}
    for label,root in (('a',a),('b',b)):
        monkeypatch.setattr(route,'_stock_triton_root',lambda root=root:root)
        route._cached_tree_digest.cache_clear()
        keys[label]=(baked_table_hash(fn,nl(11)),baked_table_hash(fn,nl(8)))
    route._cached_tree_digest.cache_clear()
    assert keys['a'][0]!=keys['b'][0]
    assert keys['a'][1]==keys['b'][1]


@pytest.mark.parametrize('version,ok',[('3.6.0',True),('3.6.1',True),('3.6.0+git9844da95',True),
                                       ('3.60.0',False),('3.7.0',False),('2.6.0',False)])
def test_exact_validated_release_family(monkeypatch,version,ok):
    import sys
    monkeypatch.setitem(sys.modules,'triton',SimpleNamespace(__version__=version,__file__='/x/triton/__init__.py'))
    if ok:
        assert route._stock_triton().__version__==version
    else:
        with pytest.raises(RuntimeError,match=r'triton==3\.6\.\*'):
            route._stock_triton()


def test_missing_triton_is_actionable_at_key_time(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules,'triton',None)
    with pytest.raises(RuntimeError,match=r'pip install "triton==3\.6\.\*"'):
        route.stock_compiler_identity()


def test_compiler_identity_is_part_of_the_stock_compile_memo(monkeypatch):
    seen=[]
    def recorder(ttir,capability,num_warps,num_stages,enable_fp_fusion=True,compiler_identity=()):
        seen.append(compiler_identity)
        return '.version 9.0\n.target sm_120a\n.visible .entry route_probe()\n{\n}\n',0,'route_probe'
    monkeypatch.setattr(route,'stock_compile',recorder)
    monkeypatch.setattr(route,'stock_compiler_identity',lambda:('stock-triton-content','3.6.0','deadbeef',()))
    fake=lambda *a,**k:[SimpleNamespace(compute_capability='12.0',platform='gpu')]
    text=_lowered(route.KernelCallCompilerParams(num_warps=4),fake,monkeypatch)
    assert 'triton_kernel_call' in text and seen==[('stock-triton-content','3.6.0','deadbeef',())]
