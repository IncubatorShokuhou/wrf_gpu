"""Gate controls on a completed twin-oracle receipt (critic F1): numpy only.

Loads the admitted production arrays, the original references and the measured
per-step floors, re-evaluates the healthy arrays (must equal the oracle verdict)
and applies pre-registered mutants to the ACTUAL side. Every mutant must FAIL
on every case-step where its term is nonzero; a mutant on an all-zero term is
reported VACUOUS (never counted as a kill).
G1 RQVBLTEN x1.5, G2 RQVBLTEN = 0, G3 RQCBLTEN sign flip (critic survivors);
positive controls Dtke x2, qBUOY x1.1, RTHBLTEN = 0, RUBLTEN sign flip, RVBLTEN x0.5;
amendment controls (critic 00:37Z) RQI x1.5 / = 0 / flip, RQC x1.5, RQV x1.1, and a
10-ulp injection at EVERY point where the per-level quantization floor binds
(each point alone; vectorized because the healthy evaluation passes).
"""
import argparse
import json
from pathlib import Path
import numpy as np
from history_gate import evaluate,quantization_floor,TKE,EDMF,PBL,QUANT

MUTANTS={'G1_RQV_x1.5':('RQVBLTEN',lambda a:a*np.float32(1.5)),'G2_RQV_zero':('RQVBLTEN',lambda a:np.zeros_like(a)),
         'G3_RQC_signflip':('RQCBLTEN',lambda a:-a),'P1_Dtke_x2':('Dtke',lambda a:a*np.float32(2)),
         'P2_qBUOY_x1.1':('qBUOY',lambda a:a*np.float32(1.1)),'P3_RTH_zero':('RTHBLTEN',lambda a:np.zeros_like(a)),
         'P4_RU_signflip':('RUBLTEN',lambda a:-a),'P5_RV_x0.5':('RVBLTEN',lambda a:a*np.float32(.5)),
         'A1_RQI_x1.5':('RQIBLTEN',lambda a:a*np.float32(1.5)),'A2_RQI_zero':('RQIBLTEN',lambda a:np.zeros_like(a)),
         'A3_RQI_signflip':('RQIBLTEN',lambda a:-a),'A4_RQC_x1.5':('RQCBLTEN',lambda a:a*np.float32(1.5)),
         'A5_RQV_x1.1':('RQVBLTEN',lambda a:a*np.float32(1.1))}


def controls(oracle,probe):
    record=json.loads((oracle/'result.json').read_text());run=json.loads((probe/'result.json').read_text())
    assert record['schema']=='wrf_gpu2.w44_budget.twin_original.v3' and record['case']==run['case'] and record['edmf']==run['edmf']
    out=dict(oracle=str(oracle),probe=str(probe),case=run['case'],edmf=run['edmf'],steps={})
    with np.load(probe/'accepted_history_and_inputs.npz') as f:data={k:f[k].copy() for k in f.files}
    for step in (1,2):
        with np.load(oracle/f'step{step}_reference.npz') as f:reference={k:f[k].copy() for k in f.files}
        with np.load(oracle/f'step{step}_floors.npz') as f:floors={k:f[k].copy() for k in f.files}
        actual={n:data[f'step{step}__{n}'] for n in TKE+EDMF+PBL}
        dt=float(floors['dt']);inputs={k:floors['x_'+k] for k in QUANT.values()}
        kw=dict(edmf=bool(run['edmf']),pbl_floor={n:floors[n] for n in PBL},inputs=inputs,dt=dt)
        healthy=evaluate(actual,reference,floors['Dtke'],**kw)['passed']
        assert healthy==record['steps'][str(step)]['passed'],'healthy re-evaluation differs from the receipt'
        rows={}
        for label,(name,mutate) in MUTANTS.items():
            if not np.count_nonzero(actual[name]):rows[label]='VACUOUS';continue
            bad=dict(actual);bad[name]=mutate(actual[name])
            result=evaluate(bad,reference,floors['Dtke'],**kw)
            rows[label]='KILLED' if not result['passed'] else 'SURVIVED'
        injection={}
        for name,key in QUANT.items():
            a=np.asarray(actual[name],np.float64);r=np.asarray(reference[name],np.float64)
            q=quantization_floor(inputs[key],r,dt)
            relative=.05*np.max(np.abs(r),axis=-1)[:,None];ensemble=np.asarray(floors[name],np.float64)[:,None]
            bound=np.maximum(np.maximum(relative,ensemble),q);binding=(q>relative)&(q>ensemble)
            m=np.maximum(np.abs(np.asarray(inputs[key],np.float64)),np.abs(np.asarray(inputs[key],np.float64)+r*dt))
            e=10*np.spacing(m.astype(np.float32)).astype(np.float64)/dt
            killed=np.abs(a+np.sign(a-r+1e-300)*e-r)>bound
            injection[name]=dict(binding_points=int(binding.sum()),killed=int((killed&binding).sum()))
        rows['I10_ulp_injection']=('VACUOUS' if not sum(v['binding_points'] for v in injection.values()) else
            'KILLED' if all(v['killed']==v['binding_points'] for v in injection.values()) else 'SURVIVED')
        out['steps'][str(step)]=dict(healthy=healthy,mutants=rows,injection=injection,
            pbl_floor_max={n:float(floors[n].max()) for n in PBL},
            pbl_ratio_max={n:max(c['max_abs']/max(c['reference_max_abs'],1e-30) for c in record['steps'][str(step)]['terms'][n]['columns']) for n in PBL})
    out['verdict']='CONTROLS_PASS' if all(s['healthy'] and 'SURVIVED' not in s['mutants'].values() for s in out['steps'].values()) else 'CONTROLS_FAIL'
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair',nargs=2,action='append',metavar=('ORACLE','PROBE'),type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    rows=[controls(o,q) for o,q in a.pair]
    result=dict(schema='wrf_gpu2.w44_budget.gate_controls.v1',scope=__doc__,pairs=rows,
                verdict='CONTROLS_PASS' if all(r['verdict']=='CONTROLS_PASS' for r in rows) else 'CONTROLS_FAIL')
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verdict=result['verdict'],steps={r['case']+'/edmf'+str(r['edmf']):{k:v['mutants'] for k,v in r['steps'].items()} for r in rows})))
    if result['verdict']!='CONTROLS_PASS':raise SystemExit(2)


if __name__=='__main__':main()
