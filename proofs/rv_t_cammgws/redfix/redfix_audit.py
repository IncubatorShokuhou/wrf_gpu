"""Independent exact-call controls for the test-only bootstrap exemption."""
import ast
import json
from pathlib import Path
import runpy
import subprocess
import sys
root=Path(sys.argv[1]);out=Path(__file__).parent
sys.path.insert(0,str(root/'src'))
ns=runpy.run_path(str(root/'tests/test_aot_cheap_key.py'))
scan=ns['_import_time_env_offenders'];exempt=ns['_IMPORT_TIME_ENV_SIDE_EFFECTS']
bootstrap='gpuwrf.io.cammg_config.bootstrap_refusal'
common='import os\nfrom gpuwrf.io.cammg_config import bootstrap_refusal as check\n'
definition=ast.parse("import os\ndef bootstrap_refusal(argv):\n    return os.getenv('GPUWRF_PROBE')\n")
def check(statement):
    return scan({'gpuwrf.io.cammg_config':definition,'gpuwrf.probe':ast.parse(common+statement)})
cases={
 'exact_call':"X = check([])",
 'argument_read':"X = check(os.getenv('GPUWRF_OTHER'))",
 'condition_read':"if os.getenv('GPUWRF_OTHER'):\n    check([])",
 'neighbor_read':"X = check([])\nY = os.getenv('GPUWRF_OTHER')",
 'unrelated_same_name':"def bootstrap_refusal(argv):\n    return os.getenv('GPUWRF_OTHER')\nX = bootstrap_refusal([])",
}
results={name:check(text) for name,text in cases.items()}
assert not results['exact_call']
assert all(results[name] for name in cases if name!='exact_call')
reason=exempt.pop(bootstrap)
baseline=check(cases['exact_call'])
assert baseline==['gpuwrf/probe.py::X'],baseline
exempt[bootstrap]=reason
# Deliberately broaden the allowance to the unrelated probe callee: the control must catch it.
exempt['gpuwrf.probe.bootstrap_refusal']='blanket-mutant'
try:
    assert check(cases['unrelated_same_name']), 'unrelated same-name helper was wrongly exempted'
except AssertionError:
    mutant={'result':'KILLED','failure':'AssertionError','setup_errors':0}
else:
    raise AssertionError('blanket exemption mutant survived')
result={'candidate':'c9de2b0b9f','parent':'673cf0823','changed_paths':subprocess.check_output(['git','diff','--name-only','673cf0823','c9de2b0b9f'],cwd=root,text=True).splitlines(),
        'exact_call_controls':results,'old_allowlist_control':baseline,'blanket_mutant':mutant}
assert result['changed_paths']==['tests/test_aot_cheap_key.py']
(out/'REDFIX_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
