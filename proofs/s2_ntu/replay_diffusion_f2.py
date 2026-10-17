"""Reproduce the original sixth-order source class on frozen full-grid inputs.

Original REAL4 is the physical source reference. Promoted REAL8 is a
diagnostic against the wider FAST0 component; neither is a full forecast.
No input is cropped, no source body is edited and no positivity repair is added.
"""
import argparse
import ast
import ctypes
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/v025/b_diff'))
import pristine_diffusion as oracle


def main():
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('build',type=Path)
    p.add_argument('out',type=Path);a=p.parse_args()
    context=json.loads((a.data/'context.json').read_text())
    for name,sha in context['arrays_sha256'].items():
        assert hashlib.sha256((a.data/name).read_bytes()).hexdigest()==sha,name
    first=np.load(a.data/'rk1.npz');last=np.load(a.data/'rk3.npz');maps=np.load(a.data/'maps.npz')
    dt=float(last['dt']);dt_rk=dt/context['rk_order']
    original4=oracle.build(a.build)
    report=dict(scope=__doc__,context=context,dt_rk=dt_rk,final_dt=dt,
                original_manifest=json.loads((a.build/'manifest.json').read_text()),rows=[])
    # Width-only diagnostic: core routines stay verbatim, compiler default
    # REAL becomes 8 and the bind(c) wrapper uses the matching C ABI.
    promoted=a.build/'oracle_r8.f90'
    promoted.write_text((a.build/'oracle.f90').read_text().replace('c_float','c_double'))
    command=['<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran','-O2','-fdefault-real-8',
             '-fdefault-double-8','-fPIC','-shared','-ffree-line-length-none','-ffp-contract=off',
             '-fcheck=bounds','-J',str(a.build),str(promoted),'-o',str(a.build/'oracle_r8.so')]
    c=subprocess.run(command,capture_output=True,text=True,timeout=60)
    assert c.returncode==0,c.stderr
    report['promoted_compile']=command
    source=inspect.getsource(oracle.run).replace('np.float32','np.float64').replace('ctypes.c_float','ctypes.c_double')
    rn={'np':np,'ctypes':ctypes,'MAPS':oracle.MAPS};exec(source,rn)
    product=ROOT/'src/gpuwrf/dynamics/explicit_diffusion.py'
    functions=[n for n in ast.parse(product.read_text()).body if isinstance(n,ast.FunctionDef)
               and n.name in ('_dflux6','wrf_sixth_order_scalar_tendf')]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),
                           *functions],type_ignores=[])
    ast.fix_missing_locations(module)
    ns={'jnp':np,'_NATIVE_DIFFUSION_FP32':False};exec(compile(module,str(product),'exec'),ns)
    for name in ('qi','Ni'):
        old=last[name+'_origin'];actual=last[name+'_result'];mass=last['mass_new']
        component=ns['wrf_sixth_order_scalar_tendf'](old,first['mu_new'],c1=last['c1'],c2=last['c2'],
            msftx=maps['msftx'],msfty=maps['msfty'],dt=dt_rk,
            diff_6th_factor=context['diff_6th_factor'],monotonic=True,specified_or_nested=True)
        for kind,library,run in (('original_REAL4',original4,oracle.run),
                                 ('promoted_REAL8',a.build/'oracle_r8.so',rn['run'])):
            sc=run(library,old,first['mu_new'],last['c1'],last['c2'],maps,name='m',dt=dt_rk,
                   factor=context['diff_6th_factor'],monotonic=True)
            only=(last['mass_old']*old+dt*sc)/mass
            without=(last['mass_old']*old+dt*(last[name+'_tendency']-sc))/mass
            row=dict(kind=kind,field=name,actual_count=int((actual<0).sum()),actual_min=float(actual.min()),
                original_sixth_only_count=int((only<0).sum()),original_sixth_only_min=float(only.min()),
                without_sixth_min=float(without.min()),source_component_max_abs_difference=float(np.abs(sc-component).max()),
                original_component_sha256=hashlib.sha256(sc.tobytes()).hexdigest())
            report['rows'].append(row)
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['rows'],indent=2))


if __name__=='__main__':main()
