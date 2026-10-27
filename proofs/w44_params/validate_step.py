"""Receipt readback never promotes CPU smoke into GPU qualification."""
import json
from pathlib import Path
import sys
p=Path(sys.argv[1]);smoke=bool(int(sys.argv[2]));kind=sys.argv[3]
r=json.loads(p.read_text())
assert r['status']=='PASS'
assert r['product_child_env']['JAX_PLATFORMS']=='cuda'
assert r['gpu_executed'] == (not smoke) and r['platform']==('cpu' if smoke else 'gpu')
if smoke:
    assert r['device_override']=={'JAX_PLATFORMS':'cpu'}
    assert r.get('replay_byte_exact') if kind=='prepare' else r['default_byte_exact'] and r['changed_active']
print('CPU MODEL SMOKE PASS' if smoke else 'GPU PREPARATION PASS',p)
