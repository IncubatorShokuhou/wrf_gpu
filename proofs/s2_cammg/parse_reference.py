"""Fail-closed capture of full public fields and optional original stage observations."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse(path):
    frames=[];frame=None;section=None;stage_record=False
    for line in Path(path).read_text().splitlines():
        parts=line.split()
        if not parts:
            continue
        if parts[0]=='BEGIN':
            assert frame is None
            frame={'case':int(parts[1]),'step':int(parts[2]),'stages':{}}
        elif line in ('INPUT','OUTPUT'):
            section={};frame[line.lower()]=section;stage_record=False
        elif parts[0]=='STAGE':
            section={};frame['stages'][parts[1]]=section;stage_record=True
        elif parts[0]=='END_STAGE':
            section=None
        elif parts[0]=='END':
            assert frame and set(frame['input'])==set(frame['output'])
            assert len(frame['input'])>=45
            frames.append(frame);frame=None;section=None
        elif section is not None:
            if not parts[0][0].isalpha():
                raise ValueError(f'Unexpected record {line[:100]}')
            assert parts[0] not in section,parts[0]
            # Warnings/logging are emitted outside the structured observations.
            if parts[0].startswith('WARNING:'):
                continue
            values=np.array([float(x.replace('D','E')) for x in parts[1:]],
                            dtype=np.float64 if stage_record else np.float32)
            if not values.size or not np.all(np.isfinite(values)):
                raise ValueError(f'Nonfinite/empty {frame["case"]}/{frame["step"]}/{parts[0]}')
            section[parts[0]]=values
    assert frame is None and len(frames) in (36,48)
    assert [(f['case'],f['step']) for f in frames]==[(i,s) for i in range(1,len(frames)//2+1) for s in (1,2)]
    return frames


def main():
    ap=argparse.ArgumentParser();ap.add_argument('raw',type=Path);ap.add_argument('output',type=Path)
    ap.add_argument('--compare',type=Path);args=ap.parse_args()
    frames=parse(args.raw)
    if args.compare:
        reference=parse(args.compare)
        assert len(frames)>=len(reference)
        for f,r in zip(frames[:len(reference)],reference,strict=True):
            for section in ['input','output']:
                assert set(f[section])==set(r[section])
                for name,a in f[section].items():
                    assert a.tobytes()==r[section][name].tobytes(),(f['case'],f['step'],section,name)
    bank={}
    for f in frames:
        prefix=f'c{f["case"]:02d}_s{f["step"]}'
        for section in ('input','output'):
            for name,a in f[section].items():bank[f'{prefix}_{section}_{name}']=a
        for stage,values in f['stages'].items():
            for name,a in values.items():bank[f'{prefix}_{stage}_{name}']=a
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**bank)
    census={n:sum(int(np.count_nonzero(f['output'][n][:24])) for f in frames)
            for n in 'qc_curr qi_curr qr_curr qs_curr nc3d ni3d nr3d ns3d'.split()}
    receipt={'frames':len(frames),'arrays':len(bank),'values':sum(a.size for a in bank.values()),
             'nonfinite':0,'output_nonzero_census':census,'public_BYTE_comparison':bool(args.compare),
             'raw_sha256':hashlib.sha256(args.raw.read_bytes()).hexdigest(),
             'asset_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
