"""Offline RADTN composition with exact native helper/caller interfaces."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from transcribe_gfdl import Emitter,read_routine,base
from transcribe_gfdl_lw import CO2,FLAT,TABLES

CONSTANTS=dict(r_d='287.',r_v='461.6',cp='7.*r_d/2.',rcp='r_d/cp',ep_2='r_d/r_v',
    pi2='2.*3.1415926',rlag='14.8125',degrad='3.141592653589793/180.',rtd='1./degrad',
    dxsd='.01',xsdmax='3.1',rhgrd='1.',t_ice='-40.',trad_ice='.5*t_ice',
    abscoef_w='800.',abscoef_i='500.',secang='-1.66',cldcoef_lw='1.5',
    abscoef_lw='secang*cldcoef_lw',qconv='.1e-3',ctaucw='abscoef_w*qconv',ctauci='abscoef_i*qconv')


class RadiationEmitter(Emitter):
    def node(self,node,depth,parent):
        text=node.text
        if not text.startswith('call '):return super().node(node,depth,parent)
        m=re.fullmatch(r'call\s+(\w+)\s*\((.*)\)',text);assert m,text
        name=m[1]
        if name not in {'radfs','ozon2d'}:return super().node(node,depth,parent)
        if node.label:self.emit(f"v['_go'] = jnp.where(v['_go']=={node.label}, 0, v['_go'])",depth)
        actual=base.split(m[2]);values=[]
        pointer={'tends(its,kts,j)':'tends','tendl(its,kts,j)':'tendl'}
        for value in actual:
            compact=re.sub(r'\s+','',value)
            if compact in pointer:
                values.append(f"v[{pointer[compact]!r}][:,:,v['j']-1]")
            else:values.append(base.expression(value,self.r)[0])
        if name=='radfs':
            callee='_radfs_arw_call'
            context="{'r1':environment['r1']}"
            signature=read_routine(self.source,'radfs').args
            assert actual[signature.index('ibeg')]=='1' and actual[signature.index('ko3')]=='1' and actual[signature.index('kalb')]=='0'
        else:
            callee='_ozon2d'
            context="{'rsin1':v['rsin1'],'rcos1':v['rcos1'],'rcos2':v['rcos2']}"
        mask=f"(({parent}) & (v['_go']==0))";temp=self.fresh()
        self.emit(f"{temp} = lax.cond({mask},lambda _a:{callee}({context},*_a),lambda _a:_a,({', '.join(values)},))",depth)
        for i,value in enumerate(actual):
            compact=re.sub(r'\s+','',value)
            if compact in pointer:
                key=pointer[compact]
                self.emit(f"v[{key!r}]=v[{key!r}].at[:,:,v['j']-1].set(jnp.where({mask},_r({temp}[{i}]),v[{key!r}][:,:,v['j']-1]))",depth)
            elif re.fullmatch(r'[a-z_]\w*(?:\(.*\))?',value):self.assign(value,f'{temp}[{i}]',mask,depth)


def main(source,driver,output,receipt):
    base.MODULE['nbly']=('i',None,'15');base.MODULE['nxsd']=('i',None,'310')
    for name,value in CONSTANTS.items():base.MODULE[name]=('r',None,value)
    for name in (*CO2,*FLAT,*TABLES,'r1','axsd','ptopc'):base.MODULE[name]=('r',('table',),None)
    base.MODULE['ltop']=('i',('table',),None)
    base.MODULE['sdprint']=('b',None,'.false.')
    base.INTR['fpvs']='fpvs_operator'
    base.INTR['mod']='jnp.fmod'
    routine=read_routine(source,'radtn')
    # These native module variables are assigned by RADTN, then passed
    # explicitly to the ozone helper. They never become traced global state.
    for key in ('rsin1','rcos1','rcos2'):routine.types[key]='r'
    emitter=RadiationEmitter(routine);emitter.source=source
    text=emitter.render()
    unresolved=routine.variables-set(routine.types)-set(routine.params)-set(base.MODULE)
    if unresolved:raise ValueError(('RADTN unresolved module names',sorted(unresolved)))
    declared_double=sorted(k for k,v in routine.types.items() if v=='d')
    assert declared_double==['ddx','eex','prod'],declared_double
    text=text.replace('def _radtn(', 'def _radtn(environment,',1)
    original=driver.read_text()
    prefix=original[:original.index('    def _lwr88(')]
    definitions=original[original.index('    def _lwr88('):original.rindex('    return _radfs')]
    # The only caller is the verified source CALL with literal flags1/1/0.
    # Avoid Python scalar conversion of JAX tracers inside that private call.
    definitions=definitions.replace('def _radfs(environment,','def _radfs_arw_call(environment,',1)
    definitions='\n'.join(line for line in definitions.splitlines() if "if int(ko3)!=1" not in line)+'\n'
    prefix=prefix.replace('def reference_factory(nlayers,co2_tables,cloud_operator):',
        'def reference_factory(nlayers,co2_tables,domain_static,cloud_operator,fpvs_operator,ozone_operator,zenith_operator):')
    prefix=prefix.replace('    _STATIC_TABLES=_TABLES','    _TABLES.update(domain_static)\n    _STATIC_TABLES=_TABLES')
    wrappers=[]
    zen=read_routine(source,'zenith')
    code='def _zenith('+', '.join(zen.args)+'):\n'
    code+='    czen,dayi,hour=zenith_operator(times,glon,glat,idat,ihrst)\n'
    code+='    return ('+', '.join(zen.args)+',)\n';wrappers.append(code)
    ozone=read_routine(source,'ozon2d')
    code='def _ozon2d(season,'+', '.join(ozone.args)+'):\n'
    code+="    qo3=ozone_operator(pozn[0],xlat[0],season['rsin1'],season['rcos1'],season['rcos2'])[None,:]\n"
    code+='    return ('+', '.join(ozone.args)+',)\n';wrappers.append(code)
    generated=prefix+definitions+'\n'
    for code in (*wrappers,text):generated+='\n'.join('    '+line for line in code.splitlines())+'\n\n'
    generated+='    return _radtn\n'
    ast.parse(generated);output.write_text(generated)
    receipt.write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),driver_sha256=hashlib.sha256(original.encode()).hexdigest(),
        arguments=routine.args,shapes=routine.shapes,types=routine.types,declared_DOUBLE=declared_double,
        generated_lines=len(generated.splitlines()),
        scope='offline RADTN reference composition; nativehelpers/operators+sourcepointerboundary explicit; no numerical/GPU/admission claim'),indent=2)+'\n')
    print('RADTN emitted',len(generated.splitlines()),'lines; nativeDOUBLE',declared_double,'qualification pending')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source','driver','output','receipt'):p.add_argument(key,type=Path)
    a=p.parse_args();main(a.source,a.driver,a.output,a.receipt)
