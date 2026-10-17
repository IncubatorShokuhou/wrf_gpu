#!/usr/bin/env python3
"""Original-RASIT5 propagation of the independently frozen thermal input allowance.

TRIB is a canopy-air temperature. Its 0.0005K allowance predates compiled
measurements (native_reference.json). BPS is built by eight REAL primitive
operations (pressure scaling/ratio, log, coefficient product, exp, reciprocal);
use an eight-input-ULP envelope. The transfer law itself is never changed.
This is a CPU column bound; it does not qualify a GPU/coupled forecast.
"""
from pathlib import Path
import argparse,ctypes,inspect,json,hashlib
import numpy as np
from literal import load
from literal_gate import operands


def main():
    ap=argparse.ArgumentParser();ap.add_argument('tree',type=Path);ap.add_argument('oracle',type=Path);ap.add_argument('output',type=Path);args=ap.parse_args()
    native=ctypes.CDLL(str(args.oracle)).rasit_oracle
    native.argtypes=[np.ctypeslib.ndpointer(dtype=np.float32,shape=(20,),flags='C_CONTIGUOUS')];native.restype=None
    manifest=json.loads((args.tree/'proofs/s2_ssib/native_reference.json').read_text())
    thermal=np.float32(manifest['tolerances']['temperature_atol_K'])
    literal=load(args.tree/'src/gpuwrf/physics/_ssib_generated.py');capture=[];original=literal._rasit5
    def record(*xs):capture.append(np.asarray(xs,np.float32));return original(*xs)
    literal._rasit5=record;names=tuple(inspect.signature(literal._ssib).parameters);rows=[]
    with np.errstate(all='ignore'),np.load(args.tree/'proofs/s2_ssib/native_reference.npz',allow_pickle=False) as f:
        for group in ('ordinary','snow','melt','snowfall','rainonsnow'):
            for case in range(1,6):
                fields=f[f'{group}_{case}_names'].tolist();data=f[f'{group}_{case}']
                for col in range(5):
                    frames=[dict(zip(fields,frame[:,col])) for frame in data];values=operands(names,frames[0],group!='ordinary')
                    for step,frame in enumerate(frames,1):
                        capture.clear();values['itime']=np.int32(step)
                        values=dict(zip(names,literal._ssib(*(values[n] for n in names))))
                        operands0=capture[-1].copy();truth=operands0.copy();native(truth)
                        assert truth[16].tobytes()==np.float32(frame['RIB']).tobytes(),(group,case,col,step,'caller mismatch')
                        def transfer(x):
                            return {'RIB':x[16],'USTAR':x[12],'FM':np.float32(.4)/x[17],'FH':np.float32(.4)/x[18],
                                    'CM':np.float32(x[12]*x[12])/np.float32(x[8]*x[8]),'UV10':x[19]}
                        transfer0=transfer(truth)
                        for key,value in transfer0.items():
                            assert np.float32(value).tobytes()==np.float32(frame[key]).tobytes(),(group,case,col,step,key)
                        field_bounds={key:0. for key in transfer0}
                        differences=[]
                        for sign_t in (-1,1):
                            for sign_b in (-1,1):
                                x=operands0.copy()
                                x[0]=np.nextafter(np.float32(x[0]+sign_t*thermal),np.float32(sign_t*np.inf))
                                for _ in range(8):x[15]=np.nextafter(x[15],np.float32(sign_b*np.inf))
                                inputs=(float(x[0]),float(x[15]));native(x)
                                assert np.isfinite(x).all()
                                outputs=transfer(x)
                                for key in field_bounds:field_bounds[key]=max(field_bounds[key],abs(float(outputs[key])-float(transfer0[key])))
                                differences.append({'trib':inputs[0],'bps':inputs[1],'rib':float(x[16]),'absolute':abs(float(x[16])-float(truth[16]))})
                        rows.append({'group':group,'case':case,'column':col+1,'step':step,'reference_rib':float(truth[16]),
                            'bound':max(x['absolute'] for x in differences),'field_bounds':field_bounds,'corners':differences})
                        if 'TA_RAW' in frame:values['ta']=values['tgeff']
    report={'kind':'independent-original-WRF-RASIT5-input-bound','thermal_allowance_K':float(thermal),'BPS_input_ULPs':8,
        'preregistered_manifest_sha256':hashlib.sha256((args.tree/'proofs/s2_ssib/native_reference.json').read_bytes()).hexdigest(),
        'original_source_sha256':'2b83b39640fd5d53859f1f281be59a9866baf97924e5d03d9964ec5d69bd5c73',
        'native_oracle':str(args.oracle),'rows':rows}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'calls':len(rows),'min_bound':min(r['bound'] for r in rows),'max_bound':max(r['bound'] for r in rows)}))

if __name__=='__main__':main()
