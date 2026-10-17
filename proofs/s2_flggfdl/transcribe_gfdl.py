"""Offline source transcription probe for GFDL's self-contained SWR93.

Emits ordinary single-column JAX reference source with arbitrary static layer
count. This is not a runtime interpreter or an operational/GPU admission.
Native routine fixtures must establish fidelity after transcription.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

import transcription_base as base

base.INTR.update(aint='jnp.trunc',log10='jnp.log10')
# The only module-scope parameter read by SWR93 (module_ra_gfdleta.F:12).
base.MODULE['rthresh']=('r',None,'1.e-15')


class Shape:
    def __init__(self,values):self.values=values
    def __repr__(self):return '('+','.join(self.values)+',)'


class Emitter(base.Emitter):
    def shape(self,dimensions):
        constants=dict(its='1',ite='1',ims='1',ime='1',ids='1',ide='2',
                       jts='1',jte='1',jms='1',jme='1',jds='1',jde='2',
                       kts='1',kms='1',kds='1',kte='_NLAYERS',
                       kme='(_NLAYERS+1)',kde='(_NLAYERS+1)')
        constants.update(self.r.params)
        constants.update({name:value for name,(_,dimensions,value) in base.MODULE.items()
                          if not dimensions and value is not None})

        def value(expression):
            for name,replacement in constants.items():
                expression=re.sub(r'\b'+name+r'\b','('+replacement+')',expression)
            # Check the generated Python expression; never eval source strings
            # in the runtime module. Only layer count and numeric operators.
            tree=ast.parse(expression,mode='eval')
            for node in ast.walk(tree):
                if not isinstance(node,(ast.Expression,ast.BinOp,ast.UnaryOp,ast.Constant,
                                        ast.Name,ast.Load,ast.Add,ast.Sub,ast.Mult,ast.Div,
                                        ast.USub,ast.UAdd,ast.Pow)):
                    raise ValueError(('unsupported shape',expression))
                if isinstance(node,ast.Name) and node.id!='_NLAYERS':
                    raise ValueError(('unresolved shape',self.r.name,expression))
            return expression

        dimensions_out=[]
        for dim in dimensions:
            if ':' in dim:
                lower,upper=dim.split(':')
                dimensions_out.append('('+value(upper)+')-('+value(lower)+')+1')
            else:dimensions_out.append(value(dim))
        return Shape(dimensions_out)

    def node(self,node,depth,parent):
        if node.label:
            # A branch to a labelled loop terminal must resume there even
            # though _go!=0 has disabled the current statement mask.
            self.emit(f"v['_go'] = jnp.where(v['_go']=={node.label}, 0, v['_go'])",depth)
            node=base.Node(node.text,node.line,0,node.children,node.other)
        if not node.text.startswith('do '):return super().node(node,depth,parent)
        # Radiation DO bounds are not SSiB's <=13. Use the exact native trip
        # count for dynamic cloud ranges and arbitrary static vertical grids.
        self.emit(f'# WRF:{node.line} {node.text}',depth)
        mask=f"(({parent}) & (v['_go']==0))"
        name,limits=node.text[3:].split('=',1)
        limits=base.split(limits)
        step=limits[2] if len(limits)==3 else '1'
        if re.fullmatch(r'[+-]?\d+',step):
            stride=int(step)
            if stride==0:raise ValueError('zero DO step')
            stride_code=f'jnp.int32({stride})'
        else:
            # RADTN's solar sampling uses the native positive NPHS operand.
            # Caller validation owns the nonzero-step precondition.
            stride_code=base.expression(step,self.r)[0]
        start,end=[base.expression(x,self.r)[0] for x in limits[:2]]
        prefix=self.fresh();count=self.fresh();func=self.fresh();index=self.fresh()
        self.emit(f'{prefix} = {mask}',depth)
        self.emit(f'{count} = jnp.maximum(jnp.int32(0), (({end})-({start}))//({stride_code})+jnp.int32(1))',depth)
        self.emit(f'def {func}({index},v):',depth)
        self.emit('v = dict(v)',depth+1)
        active=f"({prefix} & (v['_go']==0))"
        self.assign(name,f'({start})+{index}*({stride_code})',active,depth+1)
        self.body(node.children,depth+1,active)
        self.emit('return v',depth+1)
        self.emit(f'v = lax.fori_loop(0,{count},{func},v)',depth)
        self.assign(name,f'({start})+{count}*({stride_code})',f"({prefix} & (v['_go']==0))",depth)
        self.emit("v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])",depth)


def read_routine(source,name='swr93'):
    raw=source.read_text();rows=list(base.lines(raw))
    defines=set(re.findall(r'(?im)^#define\s+(\w+)',raw.lower()))
    routine=None;body=[];conditions=[]
    for number,text in rows:
        match=re.fullmatch(r'subroutine\s+'+re.escape(name)+r'\s*\((.*)\)',text)
        if match:
            routine=base.Routine(name,base.split(match[1]));continue
        if routine is not None:
            if text.startswith('end subroutine'):
                assert not conditions,'Unclosed native CPP conditional'
                routine.body=base.parsebody(body,routine);break
            if text.startswith('#ifdef '):conditions.append(text.split()[1] in defines);continue
            if text=='#else':
                assert conditions;conditions[-1]=not conditions[-1];continue
            if text=='#endif':
                assert conditions;conditions.pop();continue
            if conditions and not all(conditions):continue
            if text.startswith('#'):raise ValueError(('unsupported native CPP',text))
            body.append((number,text))
    assert routine is not None and routine.body
    return routine


def readonly_equivalence_data(routine):
    """Resolve DATA-only storage aliases; reject every runtime alias write.

    RADFS's ALB11/22/33 initialize contiguous pieces of ALBD. No runtime
    statement reads or writes these alias names. Keep the native flat order.
    """
    remaining=[];groups=[]
    for node in routine.body:
        if node.text.startswith('equivalence '):groups.extend(base.split(node.text[len('equivalence '):]))
        else:remaining.append(node)
    for group in groups:
        parts=base.split(group.strip()[1:-1]);assert len(parts)==2,group
        matches=[re.fullmatch(r'(\w+)\((.*)\)',x.strip()) for x in parts]
        assert all(matches),group
        alias,target=[m[1] for m in matches]
        if alias not in routine.data:raise ValueError(('non-DATA EQUIVALENCE',group))
        if any(re.search(r'\b'+alias+r'\b',node.text) for node in base.descend(remaining)):
            raise ValueError(('runtime DATA alias access requires explicit port',alias))
        dims=[int(x) for x in routine.shapes[target]]
        positions=[int(x) for x in base.split(matches[1][2])]
        offset=0;stride=1
        for pos,dim in zip(positions,dims):offset+=(pos-1)*stride;stride*=dim
        data=routine.data.setdefault(target,[None]*stride)
        values=routine.data[alias]
        assert offset+len(values)<=len(data)
        for i,value in enumerate(values):
            assert data[offset+i] in (None,value)
            data[offset+i]=value
    for target,values in routine.data.items():
        if any(v is None for v in values):raise ValueError(('partially initialized DATA alias',target))
    routine.body=remaining
    return groups


def main(source,output,receipt):
    routine=read_routine(source)
    rendered=Emitter(routine).render()
    unresolved=routine.variables-set(routine.types)-set(routine.params)-set(base.MODULE)
    if unresolved:raise ValueError(('unresolved native module variables',sorted(unresolved)))
    header='''"""Generated GFDL SWR93 column reference; CPU/native qualification pending.
Regenerate with proofs/s2_flggfdl/transcribe_gfdl.py. No runtime source parsing.
Pristine source SHA256: SOURCE_SHA
"""
import jax.numpy as jnp
from jax import lax

def _min(*args):
    value=args[0]
    for x in args[1:]:value=jnp.minimum(value,x)
    return value

def _max(*args):
    value=args[0]
    for x in args[1:]:value=jnp.maximum(value,x)
    return value

def _sign(a,b):return jnp.copysign(jnp.abs(a),b)
def _nint(x):return jnp.where(x>=0.,jnp.floor(x+.5),jnp.ceil(x-.5)).astype(jnp.int32)

def reference_factory(nlayers):
    _NLAYERS=int(nlayers)
    if _NLAYERS<3:raise ValueError('WRF radiation needs at least three layers')
'''.replace('SOURCE_SHA',hashlib.sha256(source.read_bytes()).hexdigest())
    generated=header+'\n'.join('    '+line for line in rendered.splitlines())+'\n    return _swr93\n'
    ast.parse(generated)
    output.write_text(generated)
    receipt.write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        routine='SWR93',arguments=routine.args,shapes=routine.shapes,types=routine.types,
        source_statements=len(list(base.descend(routine.body))),generated_lines=len(generated.splitlines()),
        source_line_start=routine.body[0].line,source_line_end=routine.body[-1].line,
        scope='offline source transcription only; no fidelity/performance/GPU claim'),indent=2)+'\n')
    print('SWR93 emitted',len(generated.splitlines()),'lines; qualification pending')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('source','output','receipt'):p.add_argument(k,type=Path)
    a=p.parse_args();main(a.source,a.output,a.receipt)
