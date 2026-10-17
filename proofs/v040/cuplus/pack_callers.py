"""Seal portable outputs of independently compiled original full WRF callers."""
from __future__ import annotations
import argparse,hashlib,json,os,re
from pathlib import Path
import numpy as np
from compare_kfcup_parcel import parse_text


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt',action='append',required=True,help='group=receipt.json')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    arrays={};rows=[];receipts={}
    for item in args.receipt:
        group,name=item.split('=',1);path=Path(name);receipt=json.loads(path.read_text())
        assert receipt['stage']=='COMPLETE'
        receipts[group]=dict(receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_sha256=receipt.get('source_sha256',receipt.get('source_object_sha256')),
            compiled_source_sha256=receipt.get('compiled_source_sha256',receipt.get('measured_cup_sha256')),
            flags=receipt['flags'],driver_sha256=receipt['driver_sha256'],
            profile_bank=receipt.get('profile_bank'),sequence=receipt.get('sequence'),
            wind_rotation_degrees=receipt.get('wind_rotation_degrees'))
        for number,entry in enumerate(receipt['records']):
            assert entry['returncode']==0
            text=Path(entry['stdout']).read_text()
            split=re.split(r'^SEQUENCE_STEP=\s*(\d+)\s*$',text,flags=re.M)
            prefix,_=parse_text(split[0]) if len(split)>1 else ({},{})
            frames=[(int(split[i]),split[i+1]) for i in range(1,len(split),2)] if len(split)>1 else [(1,text)]
            for step,body in frames:
                profiles,scalars=parse_text(body);profiles=dict(prefix,**profiles)
                key=f'{group}_{number}_{step}'
                for field,value in {**profiles,**scalars}.items():
                    value=np.asarray(value,np.float32)
                    assert np.isfinite(value).all(),(group,number,step,field)
                    arrays[key+'_'+field]=value
                rows.append(dict(group=group,key=key,step=step,case=entry['case'],
                    pbl=entry.get('pbl'),surface=entry.get('surface'),label=entry.get('label'),
                    background_wind=entry.get('background_wind'),controls=entry.get('controls'),
                    profiles=sorted(profiles),scalars=sorted(scalars),
                    original_stdout_sha256=hashlib.sha256(text.encode()).hexdigest()))
    args.output.mkdir(parents=True,exist_ok=True)
    file=args.output/'full_callers.npz';np.savez_compressed(file,**arrays)
    manifest=dict(scope=__doc__,reference='unmodified original WRF 4.7.1 full numerical bodies; independent Fortran callers',
        dtype='original REAL32 stdout values round-tripped to float32',rows=rows,receipts=receipts,
        fixture_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),values=sum(a.size for a in arrays.values()),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(frames=len(rows),values=manifest['values'],bytes=file.stat().st_size,
        groups={key:sum(row['group']==key for row in rows) for key in receipts})),flush=True)


if __name__=='__main__':main()
