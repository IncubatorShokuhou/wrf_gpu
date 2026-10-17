import hashlib
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET

P=Path(__file__).resolve().parent
def xml(name):
    cases=ET.parse(P/name).getroot().findall('.//testcase')
    failures=[c for c in cases if c.find('failure') is not None]
    errors=[c for c in cases if c.find('error') is not None]
    skipped=[c for c in cases if c.find('skipped') is not None]
    return dict(total=len(cases),passed=len(cases)-len(failures)-len(errors)-len(skipped),
        failures=len(failures),errors=len(errors),skipped=len(skipped)),failures
good,_=xml('RV25b_tests.xml')
assert good['passed']>17 and good['failures']==good['errors']==0 and good['skipped']==3,good
moist,failures=xml('RV28_moist.xml')
clamp,clampfail=xml('RV28_dzclamp.xml')
assert moist==dict(total=20,passed=1,failures=16,errors=0,skipped=3),moist
assert clamp==dict(total=1,passed=0,failures=1,errors=0,skipped=0),clamp
families=dict(NSSL=0,Ferrier=0,Goddard=0)
for c in failures+clampfail:
    failure=c.find('failure')
    # Pytest xunit2 puts the exception class in message/traceback, not @type.
    assert failure.get('message','').startswith('AssertionError:'),c.attrib
    assert 'AssertionError' in (failure.text or ''),c.attrib
for c in failures:
    match=re.search(r'\[(?:float(?:32|64)-)?(\d+)\]',c.get('name',''))
    assert match,c.attrib
    mp=int(match.group(1))
    families['NSSL' if mp in (17,18,19,21,22) else 'Ferrier' if mp in (5,15) else 'Goddard']+=1
assert families==dict(NSSL=11,Ferrier=3,Goddard=2),families
probe=json.loads((P/'RV27_compiled_probe.json').read_text())
assert len(probe['results'])==2 and len(probe['compiled_adapter_prefixes'])==13
assert probe['requested_and_actual_dtypes_verified'] and probe['platform']=='cpu' and not probe['gpu_executed']
identities={}
for case,prefix in [('swiss','f672a573'),('prod','6fc715a2')]:
    v=json.loads((P/f'RV29_{case}_identity.json').read_text())
    producer=json.loads((Path('<USER_HOME>/wrf_gpu2_lanes/t-classd')/f'identity_{case}_snap_a86a.json').read_text())
    sha=v['arms']['release']['sha']
    assert sha.startswith(prefix) and sha==producer['arms']['release']['sha']
    assert v['exec_steps']==0 and v['port_probe']['platform']=='cpu' and not v['port_probe']['gpu_executed']
    identities[case]=dict(sha=sha,registered_gate_prefix=prefix,producer_receipt_exact=True,executed_forecast_steps=0)
out=dict(head='ab5545b28fc0209229f3aca9d3230f972408da14',targeted=good,compiled_prefixes=13,
    helper_dtypes=2,moist_factor_mutant=moist,moist_assertions_by_family=families,dz_clamp_mutant=clamp,
    genuine_numeric_assertions=17,setup_errors=0,default_identities=identities,
    scope='CPU caller/WRF statement oracle, unchanged numerical-kernel oracle reuse; not full forecast/GPU/release qualification')
(P/'QUALIFICATION.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
