import sys,json,hashlib
from pathlib import Path
ROOT=Path('<USER_HOME>/wrf_gpu2_lanes/rv-t-classd/snapshot')
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/v040')]
import jax
jax.config.update('jax_enable_x64',True)
import numpy as np
from test_nssl_one_moment_radar import _load,_canonical
from gpuwrf.physics.nssl2mom.constants import get_one_moment_constants
from gpuwrf.physics.nssl2mom.diagnostics import radardd02_one_moment
from gpuwrf.physics.nssl2mom.indices import FP32
data=_load(19,'fp32',4)
an=_canonical(data,'S5').astype(np.float32)
dn=np.asarray(data['stages']['S0']['dn1'],np.float32)
C=get_one_moment_constants('fp32',19)
fn=lambda a,d:radardd02_one_moment(a,d,C,FP32)
jp=jax.make_jaxpr(fn)(an,dn)
logs=[dict(primitive=e.primitive.name,input_dtypes=[str(x.aval.dtype) for x in e.invars if hasattr(x.aval,'dtype')],
           output_dtypes=[str(x.aval.dtype) for x in e.outvars if hasattr(x.aval,'dtype')]) for e in jp.jaxpr.eqns if e.primitive.name=='log']
hlo=jax.jit(fn).lower(an,dn).compiler_ir('hlo').as_hlo_text()
(ROOT.parent/'RV08_precision.json').write_text(json.dumps(dict(logs=logs,
    hlo_log_lines=[l for l in hlo.splitlines() if ' log(' in l],
    wrf='module_mp_nssl_2mom.F:8811 REAL dtmp;:9648 dbz=Max(dbzmin,10.0*Log10(dtmp))',
    note='CPU typed lowering/source proof; no GPU timing or resource claim'),indent=2)+'\n')
print(logs)
