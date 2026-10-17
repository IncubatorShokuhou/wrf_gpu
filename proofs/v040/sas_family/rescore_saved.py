"""Rescore paid finite science with old/new gates; never rerun model kernels."""
from pathlib import Path
import ast
import json
import subprocess
import sys
import numpy as np
from evidence_gate import field_score, require_finite
from probe_ksas14 import compare_fields

lane=Path(sys.argv[1]);output=Path(sys.argv[2]);report={'model_rerun':False,'native':{},'kf':{},'helpers':{}}
def original_node(file,name,kind=ast.FunctionDef):
    source=subprocess.check_output(['git','show','1a9da66b2:proofs/v040/sas_family/'+file],text=True)
    return next(n for n in ast.parse(source).body if isinstance(n,kind) and
                (n.name==name if kind==ast.FunctionDef else isinstance(n.target,ast.Tuple)))
namespace={'np':np}
exec(compile(ast.Module(body=[original_node('probe_ksas14.py','compare_fields')],type_ignores=[]),'original_native_gate','exec'),namespace)
for scheme,arm in ((14,'SS48_native14'),(96,'SS49_native96')):
    for mode in ('r4','r8'):
        with np.load(lane/arm/'inputs.npz') as d:inp=dict(d)
        with np.load(lane/arm/f'oracle_{mode}.npz') as d:inp.update(dict(d))
        with np.load(lane/arm/f'candidate_{mode}.npz') as d:actual=dict(d)
        inp.update(DT=np.full(455,18.),STEPCU=np.ones(455))
        require_finite(**actual)
        old=namespace['compare_fields'](actual,inp,mode);new=compare_fields(actual,inp,mode)
        for field in old:
            for key in ('pass','max_abs','failed_values','max_scaled','active_oracle'):
                assert old[field][key]==new[field][key],(scheme,mode,field,key)
        report['native'][f'{scheme}_{mode}']={'fields':len(old),'all_original_score_values_exact':True,
                                            'raw_pass':{k:v['pass'] for k,v in new.items()}}
oldsource=subprocess.check_output(['git','show','1a9da66b2:proofs/v040/sas_family/probe_kf99_full.py'],text=True)
newsource=Path(__file__).with_name('probe_kf99_full.py').read_text()
def score_loop(source):
    return next(n for n in ast.parse(source).body if isinstance(n,ast.For) and
                isinstance(n.target,ast.Tuple) and [e.id for e in n.target.elts]==['name','b'])
for mode,arm in (('r4','SS92_oldKF_native'),('r8','SS90_oldKF_full')):
    with np.load(Path(__file__).parent/f'fixtures/kf99_{mode}.npz') as d:inputs=dict(d)
    with np.load(lane/arm/f'candidate_{mode}.npz') as d:actual=dict(d)
    stage=json.loads((lane/'SS78_oldKF_trigger'/f'trigger_{mode}.json').read_text())
    dtype=np.float32 if mode=='r4' else np.float64
    source_q0=np.array([stage[str(i)]['environment']['Q0'] for i in range(1,6)],dtype)
    expected={k:inputs[k] for k in ('RTHCUTEN','RQVCUTEN','RQCCUTEN','RQRCUTEN','RQICUTEN','RQSCUTEN','RAINCV','PRATEC','NCA','TIMEC')}
    results=[]
    for source in (oldsource,newsource):
        ns=dict(np=np,mode=mode,inputs=inputs,actual=actual,expected=expected,source_q0=source_q0,fields={},field_score=field_score)
        exec(compile(ast.Module(body=[score_loop(source)],type_ignores=[]),'saved_KF_score','exec'),ns)
        results.append(ns['fields'])
    old,new=results
    for field in old:
        assert old[field]['pass']==new[field]['pass'] and old[field]['max_abs']==new[field]['max_abs'],(mode,field)
        assert old[field]['bad_values']==new[field]['bad_values'],(mode,field,'bad values')
    report['kf'][mode]={'fields':len(old),'all_original_scores_exact':True,'raw_pass':{k:v['pass'] for k,v in new.items()}}
for path in sorted((lane/'SS69_oldKF_helpers').glob('candidate_*.npz')):
    with np.load(path) as d:a,b,x=d['actual'],d['expected'],d['input']
    mode=path.stem.rsplit('_',1)[1]
    bound=(32*np.finfo(np.float32).eps if mode=='r4' else 1.e-9)*np.max(np.abs(b),axis=0,keepdims=True)
    bad=np.argwhere(np.abs(a.astype(np.float64)-b)>bound).tolist()
    result=field_score(a,b,lambda:bound,operands=(x,))
    assert result['bad_values']==bad and result['pass']==(not bad)
    report['helpers'][path.stem]={'score_exact':True,'raw_pass':result['pass']}
output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
