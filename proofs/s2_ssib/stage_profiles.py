#!/usr/bin/env python3
"""Freeze the additional unmodified-WRF snow regimes and independent FMA spreads."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from audit_oracle import fields


def main():
    ap=argparse.ArgumentParser();ap.add_argument('lane',type=Path);ap.add_argument('destination',type=Path);args=ap.parse_args()
    dest=args.destination; fixture=dict(np.load(dest/'native_reference.npz',allow_pickle=False));manifest=json.loads((dest/'native_reference.json').read_text());spreads=json.loads((dest/'fma_reference_spread.json').read_text())
    groups=[('melt','SS22_melt','SS30_melt_fma'),('snowfall','SS28_snowfall','SS32_snowfall_fma'),('rainonsnow','SS29_rainonsnow','SS33_rainonsnow_fma')]
    for group,folder,fmafolder in groups:
        receipt=json.loads((args.lane/folder/'receipt.json').read_text())
        assert receipt['compiled_source_sha256']==manifest['source_sha256']
        maximum={};different=0
        for case in range(1,6):
            raw=args.lane/folder/f'case_{case}.stdout'; records=[fields(t) for t in raw.read_text().split('STEP=')[1:]]
            fma=[fields(t) for t in (args.lane/fmafolder/f'case_{case}.stdout').read_text().split('STEP=')[1:]]
            names=sorted({k.split('[')[0] for k in records[0]})
            fixture[f'{group}_{case}_names']=np.asarray(names)
            fixture[f'{group}_{case}']=np.asarray([[[d[f'{n}[{c}]'] for c in range(1,6)] for n in names] for d in records],np.float32)
            assert np.isfinite(fixture[f'{group}_{case}']).all()
            for a,b in zip(records,fma):
                assert a.keys()==b.keys()
                for key,x in a.items():
                    y=b[key];assert np.isfinite(y)
                    field=key.split('[')[0];delta=abs(x-y);different+=int(delta!=0)
                    maximum[field]=max(maximum.get(field,0.),delta)
            manifest['inputs'].append({'group':group,'case':case,'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'receipt':str(args.lane/folder/'receipt.json')})
        spreads.append({'group':group,'different_values':different,'max_abs_per_field':maximum})
    np.savez_compressed(dest/'native_reference.npz',**fixture)
    (dest/'fma_reference_spread.json').write_text(json.dumps(spreads,indent=2)+'\n')
    manifest['fixture_sha256']=hashlib.sha256((dest/'native_reference.npz').read_bytes()).hexdigest()
    manifest['fma_reference_spread_sha256']=hashlib.sha256((dest/'fma_reference_spread.json').read_bytes()).hexdigest()
    manifest['column_counts'].update(melt=25,melt_steps=6,snowfall=25,snowfall_steps=3,rainonsnow=25,rainonsnow_steps=3)
    (dest/'native_reference.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'groups':list(g[0] for g in groups),'fixture_bytes':(dest/'native_reference.npz').stat().st_size}))

if __name__=='__main__':main()
