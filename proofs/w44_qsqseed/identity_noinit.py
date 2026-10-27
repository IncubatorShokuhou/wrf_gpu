"""Schema-2 full production step probe with just MYNN init bypassed.

Pairs on base/fix prove the rest of the production native program is exact,
including all large constant VALUES and native Pallas kernels. No model source
is edited. Only the first-call helper flag is replaced by static False. The
normal restart and cycling paths are separately tested without intervention.
"""
from pathlib import Path
import hashlib
import json
import sys
root=Path(sys.argv[1]);output=Path(sys.argv[3]);case=sys.argv[2]
sys.path[:0]=[str(root/'src')]
from gpuwrf.coupling import physics_couplers as C
original=C._mynn_state_with_first_call_qke

def no_init(state,grid,first_timestep,**kw):
    return original(state,grid,False,**kw)
C._mynn_state_with_first_call_qke=no_init
argv=[str(root/'scripts/off_identity_probe.py'),'--alias',str(root),'--case',case,'--out',str(output)]
if '--nested' in sys.argv[4:]:argv.append('--nested')
sys.argv=argv
try:
    tool=root/'scripts/off_identity_probe.py'
    code=tool.read_text()
    # Upstream exits with os._exit, which skips finally. Change ONLY exit
    # plumbing in memory so the scientific receipt records our intervention.
    assert code.count('os._exit(code)')==1
    code=code.replace('os._exit(code)','raise SystemExit(code)')
    exec(compile(code,str(tool),'exec'),{'__file__':str(tool),'__name__':'__main__'})
finally:
    if output.exists():
        report=json.loads(output.read_text());report['qsqseed_intervention']={'helper_first_timestep':False,
            'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'all production work except MYNN first-call initialization'}
        output.write_text(json.dumps(report,indent=2)+'\n')
