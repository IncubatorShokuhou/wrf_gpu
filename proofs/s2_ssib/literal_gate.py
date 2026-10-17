#!/usr/bin/env python3
"""Native REAL4 BYTE comparison for the no-FMA scalar transcription."""
from pathlib import Path
import argparse,inspect,json,time
import numpy as np
from literal import load

CARRY={'www1','www2','www3','tc','tgs','td','snoa','snob','ta','bedo','isnow'}
INTEGERS={'ix','jx','itime','ivgtyp','isnow','sw_physics','mminlu'}
INPUTS={'ddtt','zlat','sunangle','ppl','ppc','rlwdown','zwind2','umm','vmm','qm','tm','pm','psur','ivgtyp','swdown1','day','sw_physics'}
ALIASES={'xhsflx':'HFX','elaten':'LH','ghtflx':'GRDFLX','xhlflx':'QFX','tgeff':'TGEFF'}


def operands(names,data,snow):
    values={}
    for name in names:
        value=0
        if name in ('ix','jx','itime'):value=1
        elif name in CARRY:value=data[name.upper()+'_IN']
        elif name in INPUTS:value=data[name.upper()]
        elif name=='cloud':value=data['CLOUD_IN']
        elif name.startswith('salb'):value=.1 if name in ('salb11','salb12') else .2
        elif name.startswith('radfrac'):value={'radfrac11':.3,'radfrac12':.1,'radfrac21':.4,'radfrac22':.2}[name]
        elif name=='tkair' and not snow:value=data['TM']
        values[name]=(np.int32 if name in INTEGERS else np.float32)(value)
    return values


def main():
    ap=argparse.ArgumentParser();ap.add_argument('tree',type=Path);ap.add_argument('output',type=Path);args=ap.parse_args()
    module=load(args.tree/'src/gpuwrf/physics/_ssib_generated.py');names=tuple(inspect.signature(module._ssib).parameters)
    begin=time.monotonic();rows=[]
    with np.errstate(all='ignore'),np.load(args.tree/'proofs/s2_ssib/native_reference.npz',allow_pickle=False) as f:
        for group in ('ordinary','snow','melt','snowfall','rainonsnow'):
            for case in range(1,6):
                fields=f[f'{group}_{case}_names'].tolist();array=f[f'{group}_{case}']
                for col in range(5):
                    frames=[dict(zip(fields,frame[:,col])) for frame in array];values=operands(names,frames[0],group!='ordinary')
                    for step,frame in enumerate(frames,1):
                        values['itime']=np.int32(step);result=module._ssib(*(values[n] for n in names));actual=dict(zip(names,result));differences=[];checked=0
                        for n,x in actual.items():
                            if n in INPUTS or n in ('ix','jx','itime','mminlu','cloud') or n.startswith('radfrac'):continue
                            key=ALIASES.get(n,n.upper())
                            if n=='ta' and 'TA_RAW' in frame:key='TA_RAW'
                            if key not in frame:continue
                            x=np.asarray(x);y=np.asarray(frame[key],x.dtype);checked+=1
                            assert np.isfinite(x).all() and np.isfinite(y).all(),(group,case,col,step,n)
                            if x.tobytes()!=y.tobytes():differences.append({'field':n,'actual':float(x),'reference':float(y),'abs':abs(float(x)-float(y))})
                        rows.append({'group':group,'case':case,'column':col+1,'step':step,'checked':checked,'different':len(differences),'differences':differences})
                        values=actual
                        if 'TA_RAW' in frame:values['ta']=actual['tgeff']
    report={'kind':'no-FMA-literal-REAL4-BYTE-vs-pristine-WRF','seconds':time.monotonic()-begin,'passed':all(r['different']==0 for r in rows),'rows':rows}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'seconds':report['seconds'],'different':sum(r['different'] for r in rows),'calls':len(rows)}))
    print([r for r in rows if r['different']][:2])
    assert report['passed'], 'original-WRF literal BYTE gate failed'

if __name__=='__main__':main()
