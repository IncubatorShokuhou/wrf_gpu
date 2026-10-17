"""Run the unchanged official release probe with lane-local temporary files."""
from pathlib import Path
import hashlib
import json
import sys

root,output=(Path(s).resolve() for s in sys.argv[1:3])
probe=root/'proofs/s1_port/release_identity.py'
source=probe.read_text()
assert source.count('<USER_HOME>/wrf_gpu2_lanes/s1-port')==1
source=source.replace('<USER_HOME>/wrf_gpu2_lanes/s1-port',str(output.parent))
exec(compile(source,str(probe),'exec'),{'__file__':str(probe),'__name__':'__main__'})
receipt=json.loads(output.read_text())
expected=('6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b'
          if 's0_case_20260725' in receipt['case'] else
          'f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d')
assert receipt['arms']['release']['sha']==expected,receipt['arms']
receipt['lane_probe']={'upstream_sha256':hashlib.sha256(probe.read_bytes()).hexdigest(),
                       'expected_full_sha':expected,'only_change':'lane-local temporary path'}
output.write_text(json.dumps(receipt,indent=2)+'\n')
