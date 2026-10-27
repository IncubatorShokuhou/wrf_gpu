"""Unmodified PUBLIC WRF driver on the admitted twin's actual two-step inputs."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np
from history_gate import evaluate,TKE,EDMF,PBL
from pristine_conditioning import measure_original


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--floors',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu' and not Path('/tmp/wrf_gpu2_quiet').exists()
    run=json.loads((args.probe/'result.json').read_text())
    assert run['verdict']=='TWIN_COMPONENT_2STEP_BYTE' and run['byte_fields']==run['total_fields']==50
    root=args.source
    helper=root/'tests/v025/t_cfix_mynn/f20_oracle.py'
    spec=importlib.util.spec_from_file_location('original_public',helper);O=importlib.util.module_from_spec(spec);spec.loader.exec_module(O)
    O.CALLER=(Path('<USER_HOME>/wrf_gpu2_lanes/w44-budget/P14_pristine_reference/export_on/caller.f90')).read_text()
    O.CALLER=O.CALLER.replace('bl_mynn_edmf=1',f"bl_mynn_edmf={run['edmf']}")
    anchor='  read(ui) u, v, w, th, t3d, p, exner, rho, dz, qv, qc, qi, qs, qni, qke, el_pbl\n'
    assert O.CALLER.count(anchor)==1
    O.CALLER=O.CALLER.replace(anchor,anchor.rstrip()+', qsq, qc_bl, qi_bl, cldfra_bl\n')
    O.FIELDS3=(*O.FIELDS3,'qsq','qc_bl','qi_bl','cldfra_bl')
    O.OUT3=(*O.OUT3,'dqke','qwt','qshear','qbuoy','qdiss','edmf_qt','edmf_thl','edmf_ent','edmf_qc','sub_thl3d','sub_sqv3d','det_thl3d','det_sqv3d')
    fixture=helper.parent/'fixtures/f20_columns.npz';meta=json.loads(fixture.with_suffix('.json').read_text())[run['case']]
    with np.load(fixture) as f:base={k.split('__',1)[1]:f[k].copy() for k in f.files if k.startswith(run['case']+'__')}
    base={k:v[:run['columns']].copy() for k,v in base.items()}
    base['qsfc']=base['qv'][:,0].copy();base['znt']=np.full_like(base['tsk'],np.float32(.1))
    with np.load(args.probe/'accepted_history_and_inputs.npz') as f:data={k:f[k].copy() for k in f.files}
    floor_record=json.loads((args.floors/'result.json').read_text());assert floor_record['members']>=20
    args.out.mkdir(parents=True,exist_ok=False)
    exe=O.build(args.out/'original')
    record=dict(schema='wrf_gpu2.w44_budget.twin_original.v3',
        pbl_rule='5% of reference column max, floor = same 25-member original spread per column (critic F1); RQV/RQC/RQI also per-level 4*spacing_f32(max(|x_in|,|x_in+r_ref*dt|))/dt (critic amendment 00:37Z)',case=run['case'],edmf=run['edmf'],steps={},
        scope='same-input original driver at each admitted production state; NOT a CPU-WRF coupled trajectory',
        floor_definition=floor_record['floor_definition'],floor_members=floor_record['members'],
        floor_sha256=hashlib.sha256((args.floors/(run['case']+'.npz')).read_bytes()).hexdigest(),
        production_byte_proof_sha256=hashlib.sha256((args.probe/'result.json').read_bytes()).hexdigest())
    mapping={'th':'theta','qke':'tke','el_pbl':'el','qni':'ni'}
    for step in (1,2):
        cols={k:v.copy() for k,v in base.items()}
        for key in O.FIELDS3:
            if key=='t3d':continue
            n=mapping.get(key,key)
            value=data[f'input{step}__{n}']
            if key=='qke':value=np.float32(2)*value
            cols[key]=value.copy()
        cols['t3d']=np.asarray(cols['th']*cols['exner'],np.float32)
        actual={n:data[f'step{step}__{n}'] for n in TKE+EDMF+PBL}
        expected,_=O.run(exe,args.out/f'step{step}',cols,meta['dt'],meta['dx'])
        reference={n:expected['dqke' if n=='Dtke' else n.lower()] for n in actual}
        # Re-measure on the ACTUAL operands at each step, including warm QSQ/
        # cloud inputs. No step-1 floor is reused for a different step-2 state.
        floor,conditioning=measure_original(O,exe,args.out/f'step{step}_conditioning',cols,
            expected['dqke'],meta['dt'],meta['dx'],nominal_pbl={n:reference[n] for n in PBL})
        pbl_floor={n:np.asarray(conditioning['pbl_floor'][n],np.float64) for n in PBL}
        inputs={key:np.asarray(cols[key],np.float32) for key in ('qv','qc','qi')}
        gate=evaluate(actual,reference,floor,edmf=bool(run['edmf']),pbl_floor=pbl_floor,inputs=inputs,dt=float(meta['dt']))
        np.savez(args.out/f'step{step}_floors.npz',Dtke=floor,**pbl_floor,dt=np.float64(meta['dt']),
                 **{'x_'+key:value for key,value in inputs.items()})
        gate['conditioning']=conditioning
        gate['floor_input_scope']='same actual original inputs at this step; 25 original-only members'
        record['steps'][str(step)]=gate
        np.savez(args.out/f'step{step}_reference.npz',**reference)
        (args.out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    record['verdict']='ORIGINAL_TERM_GATE_PASS' if all(s['passed'] for s in record['steps'].values()) else 'ORIGINAL_TERM_GATE_FAIL'
    (args.out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(verdict=record['verdict'],failed={k:[n for n,v in s['terms'].items() if not v['passed']] for k,s in record['steps'].items()})))
    assert not any(k=='jax' or k.startswith('gpuwrf') for k in sys.modules)
    if record['verdict']!='ORIGINAL_TERM_GATE_PASS':raise SystemExit(2)


if __name__=='__main__':main()
