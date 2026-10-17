"""Genuine assertion check of the old SSiB field selection, with source restoration."""
from pathlib import Path
import json, subprocess, sys, xml.etree.ElementTree as ET

root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve()
path=root/'src/gpuwrf/runtime/operational_mode.py'
old='ssib_rad=tuple(jnp.asarray(x,jnp.float32) for x in (rdiag.swdown,rdiag.glw,rdiag.coszen))'
new='ssib_rad=tuple(jnp.asarray(x,jnp.float32) for x in (rdiag.swnorm,rdiag.glw,rdiag.coszen))'
original=path.read_bytes();text=original.decode();assert text.count(old)==1
try:
    path.write_text(text.replace(old,new))
    run=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
        '--basetemp',str(out/'pytest/mutant'),f'--junitxml={out}/mutant.xml',
        'tests/test_ssib_swdown_caller.py::test_ssib_surface_step_receives_real_daytime_swdown',
        'tests/test_t_cfix_pblmix_f23.py::test_ssib_receives_unadjusted_swdown_at_slope_rad_0'],
        cwd=root,capture_output=True,text=True,timeout=240)
    (out/'mutant_pytest.log').write_text(run.stdout+run.stderr)
    xml=ET.parse(out/'mutant.xml').getroot();fails=list(xml.iter('failure'));errors=list(xml.iter('error'))
    genuine=run.returncode==1 and len(fails)==2 and not errors and all('AssertionError' in (f.get('message','')+(f.text or '')) for f in fails)
    result=dict(mutant='SSiB SWDOWN -> swnorm',returncode=run.returncode,assertion_failures=len(fails),
        setup_errors=len(errors),genuine_kill=genuine,messages=[f.get('message','')[:180] for f in fails])
finally:
    path.write_bytes(original)
assert path.read_bytes()==original
result['source_restored']=True
(out/'mutant.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2));assert genuine
