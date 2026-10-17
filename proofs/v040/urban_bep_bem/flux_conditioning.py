"""WRF-only operand budgets and controls; never reads a candidate result.

Implements the pending CONDITIONING_REQUEST's eight-epsilon wall-flux term.
It does not replace a registered gate or declare qualification. The calibrated
WRITE-only observer must preserve all 87 original output arrays first.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def wall_value(row,bem,perturb=False,window=False):
    T=np.float32;pt=T(row['pt']);tw=T(row['ptwin' if window else 'ptw'])
    if perturb:tw=np.nextafter(tw,T(np.inf))
    hc=T(row['hcwin' if window else 'hc']);rho=T(row['da']);cp=T(1004.)
    if bem:return T(hc*T(pt-tw))
    rhs=T(T(T(hc*tw)/rho)/cp)-T(T(T(hc/rho)/cp)*pt)
    return T(T(-rho*cp)*rhs)


def budget(row,bem,window=False):
    T=np.float32;pt=T(row['pt']);tw=T(row['ptwin' if window else 'ptw'])
    hc=T(row['hcwin' if window else 'hc']);rho=T(row['da']);cp=T(1004.)
    if bem:scale=abs(float(hc)*float(pt))+abs(float(hc)*float(tw))
    else:
        left=T(T(T(hc*tw)/rho)/cp);right=T(T(T(hc/rho)/cp)*pt)
        scale=(abs(float(left))+abs(float(right)))*float(rho)*float(cp)
    return 8*np.finfo(T).eps*scale


def run(operands,save,out):
    staged=json.loads(operands.read_text());rows={};bounds={}
    for arm,receipt in staged['arms'].items():
        assert receipt['original_arrays_BYTE']==87
        bem=arm.startswith('urban3');data=np.load(save/(arm.replace('_ac','_fp32_ac')+'.npz'))
        fields=['SFW1_URB3D','SFW2_URB3D']+(['SFWIN1_URB3D','SFWIN2_URB3D'] if bem else [])
        calculated={k:np.zeros_like(data[k]) for k in fields};allowed={k:np.zeros_like(data[k],dtype=np.float64) for k in fields}
        injected={k:np.zeros_like(data[k]) for k in fields}
        calls=iter(receipt['operands']['BEMWALL' if bem else 'BEPWALL']);count=0
        for regime in range(6):
          for step in range(1,7):
            for direction in range(2):
              for building,floors in enumerate((2,3,4) if bem else (6,)):
                for level in range(floors):
                  for side in range(2):
                    row=next(calls);count+=1;assert int(row['call'])==count
                    index=building*36+level*2+direction
                    for window in ([False,True] if bem else [False]):
                        name=fields[(2 if window else 0)+side];at=(regime,step,index)
                        calculated[name][at]=wall_value(row,bem,window=window)
                        allowed[name][at]=budget(row,bem,window=window)
                        injected[name][at]=wall_value(row,bem,perturb=True,window=window)
        assert next(calls,None) is None
        summary={}
        for name in fields:
            # This joins the independent observation to the actual saved fields;
            # wrong order or an observer that changes physics is rejected.
            np.testing.assert_array_equal(calculated[name],data[name])
            active=allowed[name]>0
            error=abs(injected[name].astype(np.float64)-calculated[name])
            assert np.all(error<=allowed[name]),name
            killed=int(np.sum(2*abs(calculated[name].astype(np.float64))>allowed[name]))
            assert killed>0,name
            summary[name]={'active':int(active.sum()),'budget_min':float(allowed[name][active].min()),
                'budget_max':float(allowed[name].max()),'one_input_ULP_max_ratio':float(np.max(error[active]/allowed[name][active])),
                'heat_sign_mutant_violations':killed}
            bounds[arm+'__'+name]=allowed[name]
        rows[arm]=summary
    np.savez_compressed(out.with_suffix('.npz'),**bounds)
    out.write_text(json.dumps({'status':'REFERENCE_OPERAND_CONTROLS_ONLY_GATE_UNCHANGED',
        'operands_sha256':hashlib.sha256(operands.read_bytes()).hexdigest(),
        'source_sha256':staged['original_sha256'],'formula':'8*eps32*sum(abs(WRF subtraction operands)) with WRF unit conversion',
        'arms':rows},indent=2)+'\n')
    print(json.dumps(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--operands',type=Path,required=True)
    p.add_argument('--save',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.operands,a.save,a.out)
