"""Record compact tracked WRF oracle units/shapes/tolerances and SHA256s."""
import hashlib,json
from pathlib import Path

import numpy as np
import yaml

repo=Path(__file__).resolve().parents[2]
root=repo/'proofs/u2_tslist/oracle'
source=json.loads((root/'manifest.json').read_text())
surface_units=['K','kg kg-1','m s-1','m s-1','Pa','W m-2','W m-2','W m-2','W m-2','K','K','mm','mm','kg m-2']
surface=['T2','Q2','U10','V10','PSFC','GLW','GSW','HFX','LH','TSK','TSLB','RAINC','RAINNC','CLW']
profile=['UU','VV','WW','PH','TH','QV','PR']
units=surface_units+['m s-1','m s-1','m s-1','m','K','kg kg-1','Pa']
variables=[dict(name=name,units=unit,shape=[4,2] if name in surface else [4,2,5],dtype='float32',
                absolute_tolerance=5.1e-6,relative_tolerance=2.e-7,
                staggering='station mass point' if name not in ['UU','VV'] else 'WRF staggered, or WRF rotated-before-average option')
           for name,unit in zip(surface+profile,units)]
variables.append(dict(name='active_CLW',units='kg m-2',shape=[2],dtype='float32',absolute_tolerance=0.,
                      relative_tolerance=float(8*np.finfo(np.float32).eps),staggering='full44level pressure-face integral',
                      scope='all5species constructed active;pristine literal Fortran source on real Swiss pressures/heights'))
files={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
       for path in sorted(root.rglob('*')) if path.is_file() and path.name!='fixture.yaml'}
manifest=dict(fixture_id='u2-tslist-swiss-four-step-v1',source='Original CPU-WRF Swiss RD11 executable, private lane run',
              source_commit='UNKNOWN installed WRF commit; pinned executable SHA256 '+source['executable_sha256'],
              scenario='Swiss42x42x44 dt18 4steps,2inside+1outside station;LATLON+IJ,staggered+unstaggered,tsbuf3',
              created_utc='2026-10-13T11:51:55Z',variables=variables,files=files,
              license_notes='WRF-derived diagnostic samples; no external station truth or ALISIOS observation values',
              generation_commands=['python proofs/u2_tslist/stage_oracle.py <private-case>',
                                   '(cd <private-case> && mpirun -np 1 --bind-to none ./wrf.exe)',
                                   'python proofs/u2_tslist/capture_oracle.py <private-case> <fixture>',
                                   'python proofs/u2_tslist/stage_oracle.py <private-IJ-case> --ij-unstagger',
                                   'python proofs/u2_tslist/active_oracle.py <lane-artifacts>'],
              qualification='CPU same-input diagnostic/format only; GPU/coupled forecast pending',
              tracked_asset_authority='Binding _WAVE_S1_COMMON.md requires tracked test assets and clean archive; all fixtures <200KB, no full WRF outputs')
(root/'fixture.yaml').write_text(yaml.safe_dump(manifest,sort_keys=False))
print(json.dumps({'files':len(files),'variables':len(variables),'path':str(root/'fixture.yaml')}))
