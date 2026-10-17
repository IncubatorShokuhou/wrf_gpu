"""Generate the actual WRF-ARW RADFS caller, preserving its fixed KO3=1.

RADTN always calls RADFS with KO3=1, KALB=0, IBEG=1. The legacy KO3=0
O3INT path is outside this caller contract; fail closed rather than synthesize
that routine. Read-only EQUIVALENCE DATA initializes Payne ocean albedo.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from transcribe_gfdl import Emitter,read_routine,readonly_equivalence_data,base
from transcribe_gfdl_lw import CO2,FLAT,TABLES,ROUTINES


def main(source,lw_template,output,receipt):
    base.MODULE['nbly']=('i',None,'15')
    for name in (*CO2,*FLAT,*TABLES,'r1'):base.MODULE[name]=('r',('table',),None)
    routine=read_routine(source,'radfs')
    aliases=readonly_equivalence_data(routine)
    # Verified binding at pristine RADTN:1376-1378. Remove only the unreachable
    # legacy branch, and enforce KO3=1 at the generated function boundary.
    begin=[i for i,n in enumerate(routine.body) if re.sub(r'\s+','',n.text)=='if(ko3.gt.0)goto135']
    end=[i for i,n in enumerate(routine.body) if n.label==135]
    assert len(begin)==len(end)==1 and begin[0]<end[0]
    routine.body=routine.body[:begin[0]]+routine.body[end[0]+1:]
    text=Emitter(routine).render()
    unresolved=routine.variables-set(routine.types)-set(routine.params)-set(base.MODULE)
    if unresolved:raise ValueError(('RADFS unresolved module variables',sorted(unresolved)))
    text=text.replace('def _radfs(', 'def _radfs(environment,',1)
    lines=text.splitlines()
    lines[1:1]=[
        "    if int(ko3)!=1 or int(ibeg)!=1 or int(kalb)!=0:raise ValueError('WRF-ARW column caller requires KO3=1/IBEG=1/KALB=0')",
        '    _TABLES=dict(_STATIC_TABLES)',
        '    _TABLES.update(environment)',
    ]
    text='\n'.join(lines)
    lw=lw_template.read_text()
    prefix=lw[:lw.index('    def _lwr88(')]
    definitions=lw[lw.index('    def _lwr88('):lw.rindex('    return _lwr88')]
    sw_path=lw_template.with_name('generated_swr93.py')
    sw=sw_path.read_text()
    definitions+='\n'+sw[sw.index('    def _swr93('):sw.rindex('    return _swr93')]
    prefix=prefix.replace('Generated GFDL LWR88 family column reference, qualification pending.',
        'Generated GFDL WRF-ARW RADFS column reference, qualification pending.')
    prefix=prefix.replace('def reference_factory(nlayers,co2_tables):','def reference_factory(nlayers,co2_tables,cloud_operator):')
    prefix+='    _STATIC_TABLES=_TABLES\n'
    cloud=read_routine(source,'clo89')
    wrapper='def _clo89('+', '.join(cloud.args)+'):\n'
    wrapper+='    cldfac=cloud_operator(camt[0],kbtm[0],ktop[0],nclds[0])[None,:,:]\n'
    wrapper+='    return ('+', '.join(cloud.args)+',)\n'
    generated=prefix+definitions+'\n'+'\n'.join('    '+line for line in wrapper.splitlines())+'\n\n'
    generated+='\n'.join('    '+line for line in text.splitlines())+'\n    return _radfs\n'
    ast.parse(generated)
    output.write_text(generated)
    receipt.write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        lw_component_sha256=hashlib.sha256(lw.encode()).hexdigest(),
        sw_component_sha256=hashlib.sha256(sw.encode()).hexdigest(),
        generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        routine='RADFS',arguments=routine.args,shapes=routine.shapes,types=routine.types,
        readonly_equivalence=aliases,generated_lines=len(generated.splitlines()),
        fixed_caller=dict(ko3=1,ibeg=1,kalb=0),dynamic_environment=('r1',),
        scope='actual WRF-ARW caller only; legacyKO3=0 failclosed; qualified LW/CLO89 components reused; native caller qualification pending'),indent=2)+'\n')
    print('WRF-ARW RADFS emitted',len(generated.splitlines()),'lines; qualification pending')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source','lw_template','output','receipt'):p.add_argument(key,type=Path)
    a=p.parse_args();main(a.source,a.lw_template,a.output,a.receipt)
