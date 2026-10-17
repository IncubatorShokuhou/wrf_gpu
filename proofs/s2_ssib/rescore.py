#!/usr/bin/env python3
"""Host-only scoring of retained typed model results; never rerun an unchanged model."""
import argparse,ast,hashlib,json,subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('tree',type=Path);ap.add_argument('prefix',type=Path);ap.add_argument('bounds',type=Path);ap.add_argument('output',type=Path);ap.add_argument('--measurement-commit',default='5d591bfcb');args=ap.parse_args()
    source=args.tree/'tests/test_s2_ssib.py';tree=ast.parse(source.read_text())
    keep=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(n,ast.Name) and n.id in ('ALIASES','TEMPERATURE','WATER','FLUX') for n in node.targets):keep.append(node)
        if isinstance(node,ast.FunctionDef) and node.name=='_check':keep.append(node)
    namespace={'np':np};exec(compile(ast.Module(keep,[]),str(source),'exec'),namespace)
    root=args.tree/'proofs/s2_ssib'
    spread={r['group']:r['max_abs_per_field'] for r in json.loads((root/'fma_reference_spread.json').read_text())}
    bounds={(r['group'],r['case'],r['column'],r['step']):r.get('field_bounds',{'RIB':r['bound']}) for r in json.loads(args.bounds.read_text())['rows']}
    checked=0;failures=[];maxima={};calls=0;proof={}
    # These are the only product modules called by the scalar component arm.
    for relative in ('src/gpuwrf/physics/lsm_ssib.py','src/gpuwrf/physics/_ssib_generated.py'):
        old=subprocess.run(['git','show',args.measurement_commit+':'+relative],capture_output=True,check=True).stdout
        current=(args.tree/relative).read_bytes();assert old==current,relative
        proof[relative]=hashlib.sha256(current).hexdigest()
    with np.load(root/'native_reference.npz',allow_pickle=False) as fixture:
        for group in ('ordinary','snow','melt','snowfall','rainonsnow'):
            if not args.prefix.with_name(args.prefix.name+'_'+group+'.npz').exists():continue
            with np.load(args.prefix.with_name(args.prefix.name+'_'+group+'.npz'),allow_pickle=False) as actual:
                for row,(case,col,step) in enumerate(actual['records']):
                    calls+=1;names=fixture[f'{group}_{case}_names'].tolist()
                    expected=dict(zip(names,fixture[f'{group}_{case}'][step-1,:,col-1]))
                    for name in actual.files:
                        if name=='records':continue
                        key=namespace['ALIASES'].get(name,name.upper())
                        if name=='ta' and 'TA_RAW' in expected:key='TA_RAW'
                        if key not in expected:continue
                        checked+=1;value=actual[name][row];reference=expected[key]
                        maxima[name]=max(maxima.get(name,0.),abs(float(value)-float(reference)))
                        namespace['STATE_FIELDS']=(name,)
                        try:namespace['_check'](SimpleNamespace(**{name:value}),expected,
                            raw_ta='TA_RAW' in expected,contraction_spread=spread[group],input_bounds=bounds[group,int(case),int(col),int(step)])
                        except AssertionError as error:failures.append({'group':group,'case':int(case),'column':int(col),'step':int(step),'field':name,'message':str(error)})
    report={'kind':'host-only-exact-source-reuse-original-WRF-scoring','calls':calls,'checked':checked,
        'passed':not failures,'failures':failures,'max_abs_per_field':maxima,'product_byte_identity':proof,
        'bound_sha256':hashlib.sha256(args.bounds.read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'calls':calls,'checked':checked,'passed':not failures,'failed_fields':sorted({r['field'] for r in failures})}))

if __name__=='__main__':main()
