"""Original phase-mediation and NCA-branch witnesses on the composed old-KF core."""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
from evidence_gate import field_score, require_finite
import jax
from probe_ksas14 import parse
from gpuwrf.physics.cumulus_kf99 import kf99_column

jax.config.update('jax_enable_x64',True)
assert jax.devices()[0].platform=='cpu'
work=Path(sys.argv[1]);mode=sys.argv[2];dtype=np.float32 if mode=='r4' else np.float64
manifest=json.loads((work/'manifest.json').read_text())
report={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'mode':mode,'variants':{}}
cache={}
for name,meta in manifest['arms'][mode]['variants'].items():
    row=parse(work/f'{name}_{mode}.txt');flag=meta['phase_flag']
    require_finite(**row, nca_input=meta['nca_input'])
    if flag not in cache:
        def factory(flag):
            return jax.jit(lambda *a:kf99_column(*a[:8],dtype(54.),dtype(9000.),pi_phy=a[8],nca=a[9],
                warm_rain=flag=='warm',qi_flag=flag not in ('warm','wsm3'),qs_flag=flag not in ('warm','wsm3','nosnow'),
                qr_flag=flag!='norain',mode=mode))
        cache[flag]=factory(flag)
    out=cache[flag](*[row[k].astype(dtype) for k in ('T','QV','P','DZ','RHO','W0AVG','U','V','PI')],dtype(meta['nca_input']))
    fields={}
    for key in ('RTHCUTEN','RQVCUTEN','RQCCUTEN','RQRCUTEN','RQICUTEN','RQSCUTEN','RAINCV','PRATEC','NCA','TIMEC'):
        a=np.asarray(out[key]);b=row[key].astype(dtype)
        fields[key]=field_score(a,b,lambda: 0. if key in ('NCA','TIMEC') else
            (1.e-9 if mode=='r8' else 32*np.finfo(np.float32).eps)*np.max(np.abs(b)))

    report['variants'][name]={'fields':fields,'pass':all(v['pass'] for v in fields.values())}
    (work/f'comparison_{mode}.json').write_text(json.dumps(report,indent=2));print(name,[k for k,v in fields.items() if not v['pass']],flush=True)
