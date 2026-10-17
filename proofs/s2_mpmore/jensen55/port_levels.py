"""Offline translation of original level processes and final size checks.

Adaptive column sedimentation stays separate. Category loops keep ascending
order; the allocated list of present habits becomes three ordered masked arms.
This produces reviewable Python, never a runtime Fortran interpreter.
"""
from pathlib import Path
import ast,importlib.util,re

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('p3_lexical',ROOT/'proofs/s2_p3/port_processes.py')
lex=importlib.util.module_from_spec(spec);spec.loader.exec_module(lex)
SOURCE=Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_mp_jensen_ishmael.F')
lines=list(lex.logical_lines(SOURCE.read_text()))
CHECKPOINTS={min(n for n,s in lines if n>=start):stage for stage,start in enumerate((882,1550,1634,2420,2727))}


def split(text):
    result=[];depth=0;start=0
    for i,c in enumerate(text):
        depth+=(c=='(')-(c==')')
        if c==',' and depth==0:result.append(text[start:i].strip());start=i+1
    return result+[text[start:].strip()]


metadata={};parameters={}
for n,s in lines:
    if not 474<=n<681 or '::' not in s:continue
    decl,values=s.split('::',1)
    kind=decl.split(',')[0].strip()
    if kind not in ('real','integer','logical'):continue
    dimension=re.search(r'dimension\(([^)]*)\)',decl)
    for item in split(values):
        if '=' in item:
            name,rhs=map(str.strip,item.split('=',1));parameters[name]=rhs;continue
        m=re.fullmatch(r'(\w+)(?:\(([^)]*)\))?',item);assert m,(n,item)
        name,dims=m.groups();dims=dims or (dimension[1] if dimension else '')
        if name=='icearray':continue  # Replaced by three ordered masked habits.
        axes=split(dims) if dims else []
        trailing=tuple(a for a in axes if a!='kts:kte')
        assert not trailing or trailing[0]=='cat',(n,name,axes)
        metadata[name]=(kind,trailing)
parameters.update(pi='3.14159265',g_home='9.8',rd='287.15',rv='461.5',cp='1004.',cpw='4218.',
                  rcp='.285856574',rhow='1000.',t0='273.15',p0='101325.',rhoi='920.',r0='1.27494')
static={'cat','ice1','ice2','ice3','cc','iti','riming','rain_ice','freeze_qc','splinters','wet_grow'}
tables={'gamma_tab','itab','itabr','igrdata','coltab','coltabn'}
lex.RATES={k for k,(_,dims) in metadata.items() if len(dims)==1};lex.CATS=set()
lex.MATRICES={k for k,(_,dims) in metadata.items() if len(dims)==2}
lex.LOOPS={'cc','iti'};lex.STATIC=static;lex.TABLES=tables


class Expression(lex.Expression):
    def __init__(self,cap=None):
        super().__init__();self.cap=cap

    @staticmethod
    def folded(single,double):
        choice=ast.IfExp(test=ast.Compare(left=ast.Attribute(value=ast.Subscript(value=ast.Name(id='v',ctx=ast.Load()),slice=ast.Constant('qv'),ctx=ast.Load()),attr='dtype',ctx=ast.Load()),
                                         ops=[ast.Eq()],comparators=[ast.Constant('float32')]),body=ast.Constant(single),orelse=ast.Constant(double))
        return ast.Call(func=ast.Name(id='_r',ctx=ast.Load()),args=[choice],keywords=[])

    def visit_BinOp(self,node):
        if isinstance(node.op,ast.Pow):
            value=node.right
            if isinstance(value,ast.UnaryOp) and isinstance(value.op,ast.USub) and isinstance(value.operand,ast.Constant):
                value=ast.Constant(-value.operand.value)
            if isinstance(value,ast.Constant) and isinstance(value.value,(int,float)) and float(value.value).is_integer():
                count=int(value.value)
                if count==4 and isinstance(node.left,ast.Name) and node.left.id=='lamr' and self.cap:
                    import numpy as np
                    denominator={'lamminr':2800.e-6,'lammaxr':20.e-6}[self.cap]
                    single=float(np.float32(float(np.float32(np.float32(1.)/np.float32(denominator)))**4))
                    double=(1./denominator)**4
                    return self.folded(single,double)
                if isinstance(node.left,ast.Constant) and isinstance(node.left.value,float):
                    # Literal powers are compiler-folded with one final REAL
                    # rounding. Variable integer powers use multiply chains.
                    import numpy as np
                    single=float(np.float32(float(np.float32(node.left.value))**count))
                    double=float(node.left.value**count)
                    return self.folded(single,double)
                def power(n):
                    if n==0:return ast.Call(func=ast.Name(id='_r',ctx=ast.Load()),args=[ast.Constant(1.)],keywords=[])
                    if n==1:return self.visit(node.left)
                    if n%2==0:return ast.BinOp(left=power(n//2),op=ast.Mult(),right=power(n//2))
                    return ast.BinOp(left=power(n-1),op=ast.Mult(),right=self.visit(node.left))
                if count<0:return ast.BinOp(left=power(0),op=ast.Div(),right=power(-count))
                return power(count)
            return ast.Call(func=ast.Attribute(value=ast.Name(id='xp',ctx=ast.Load()),attr='power',ctx=ast.Load()),
                            args=[self.visit(node.left),self.visit(node.right)],keywords=[])
        return super().visit_BinOp(node)
    def visit_Name(self,node):
        if node.id in tables:return ast.Subscript(value=ast.Name(id='tables',ctx=ast.Load()),slice=ast.Constant(node.id),ctx=ast.Load())
        return super().visit_Name(node)

    def visit_Call(self,node):
        name=node.func.id
        if name=='tanh':
            return ast.Call(func=ast.Attribute(value=ast.Name(id='xp',ctx=ast.Load()),attr='tanh',ctx=ast.Load()),
                            args=[self.visit(a) for a in node.args],keywords=[])
        if name=='size':
            assert len(node.args)==2
            return ast.Subscript(value=ast.Attribute(value=self.visit(node.args[0]),attr='shape',ctx=ast.Load()),
                                 slice=ast.BinOp(left=self.visit(node.args[1]),op=ast.Sub(),right=ast.Constant(1)),ctx=ast.Load())
        if name in ('gamma','gammln','polysvp','capacitance_gamma'):
            return ast.Call(func=ast.Name(id=name,ctx=ast.Load()),args=[self.visit(a) for a in node.args],
                            keywords=[ast.keyword(arg='xp',value=ast.Name(id='xp',ctx=ast.Load()))])
        if name=='get_igr':
            return ast.Call(func=ast.Name(id=name,ctx=ast.Load()),args=[self.visit(node.args[1])],
                            keywords=[ast.keyword(arg='data',value=self.visit(node.args[0])),ast.keyword(arg='xp',value=ast.Name(id='xp',ctx=ast.Load()))])
        if name in ('nint','int','float','real'):
            fn={'nint':'_nint','int':'_int','float':'_r','real':'_r'}[name]
            return ast.Call(func=ast.Name(id=fn,ctx=ast.Load()),args=[self.visit(a) for a in node.args],keywords=[])
        return super().visit_Call(node)

    def array(self,name,args):
        assert name in metadata or name in tables,('Unknown Fortran array/intrinsic',name)
        if args and isinstance(args[-1],ast.Name) and args[-1].id=='k':args=args[:-1]
        value=self.visit_Name(ast.Name(id=name,ctx=ast.Load()))
        if not args:return value
        indices=[ast.BinOp(left=self.visit(a),op=ast.Sub(),right=ast.Constant(1)) for a in args]
        if name in tables:
            return ast.Subscript(value=value,slice=ast.Tuple(elts=indices,ctx=ast.Load()),ctx=ast.Load())
        return ast.Call(func=ast.Name(id='_get',ctx=ast.Load()),args=[value,ast.Tuple(elts=indices,ctx=ast.Load())],keywords=[])


def expr(s,cap=None):return ast.unparse(ast.fix_missing_locations(Expression(cap).visit(lex.parse_expr(s)))).replace('jnp.','xp.')


def target(s):
    node=lex.parse_expr(s)
    if isinstance(node,ast.Name):return node.id,'None'
    assert isinstance(node,ast.Call)
    args=node.args
    if args and isinstance(args[-1],ast.Name) and args[-1].id=='k':args=args[:-1]
    return node.func.id,('('+','.join('('+expr(ast.unparse(a))+')-1' for a in args)+',)' if args else 'None')


def generate(phase,start,end):
    statements=[(n,s) for n,s in lines if start<=n<=end and n!=719 and not 734<=n<=743 and n!=886]
    out=[];indent=1;stack=[];mask='True';counter=0;assigned=set();previous_source=None
    def emit(s):out.append('    '*indent+s)
    def assign(lhs,rhs,n,active):
        name,index=target(lhs);assigned.add(name)
        emit(f'_put(v,{name!r},{index},{rhs},{active},xp)  # WRF:{n}')
    for n,s in statements:
        if phase=='processes' and n in CHECKPOINTS:
            stage=CHECKPOINTS[n]
            emit(f'if observer is not None: observer({stage},v,{mask})  # Original checkpoint WRF:{n}')
        if s.startswith(('enddo','end do')):
            kind,parent,_=stack.pop();assert kind in ('level','for')
            if kind=='for':indent-=1
            mask=parent;continue
        if s in ('endif','end if'):
            kind,parent,_=stack.pop();assert kind=='if';mask=parent;continue
        if s.startswith(('else','elseif')):
            kind,parent,taken=stack[-1];assert kind=='if'
            counter+=1;mask=f'mask_{counter}'
            if s=='else':emit(f'{mask}=xp.logical_and({parent},xp.logical_not({taken}))')
            else:
                condition=re.fullmatch(r'(?:else\s*if|elseif)\s*\((.*)\)\s*then',s)[1]
                emit(f'{mask}=xp.logical_and(xp.logical_and({parent},xp.logical_not({taken})),{expr(condition)})')
                emit(f'{taken}=xp.logical_or({taken},{mask})')
            continue
        if s.startswith('do '):
            name,bounds=re.fullmatch(r'do\s+(\w+)\s*=\s*(.*)',s).groups()
            if name=='k':stack.append(('level',mask,None));continue
            assert name in ('cc','ccvar','iti'),(n,s)
            parent=mask
            if name=='ccvar':
                emit('v["ccvar"]=xp.zeros_like(v["numice"])')
                emit('for cc in range(1,4):');indent+=1;counter+=1;mask=f'mask_{counter}'
                emit(f'{mask}=xp.logical_and({parent},_get(v["has_ice"],(cc-1,)))')
                emit(f'v["ccvar"]=xp.where({mask},v["ccvar"]+1,v["ccvar"])  # Original active-list ordinal')
            else:
                a,b=bounds.split(',');emit(f'for {name} in range({expr(a)},{expr(b)}+1):');indent+=1
            stack.append(('for',parent,None));continue
        if s.startswith('if'):
            match=re.fullmatch(r'if\s*\((.*)\)\s*then',s)
            if match:
                counter+=1;child=f'mask_{counter}';taken=f'taken_{counter}'
                emit(f'{child}=xp.logical_and({mask},{expr(match[1])})');emit(f'{taken}={child}')
                stack.append(('if',mask,taken));mask=child;continue
            depth=0;last=None
            for pos,c in enumerate(s[s.index('('):],s.index('(')):
                depth+=(c=='(')-(c==')')
                if depth==0:last=pos;break
            assert last is not None
            cond,action=s[s.index('(')+1:last],s[last+1:].strip();lhs,rhs=map(str.strip,action.split('=',1))
            assign(lhs,expr(rhs),n,f'xp.logical_and({mask},{expr(cond)})');continue
        if s.startswith('call '):
            node=lex.parse_expr(s[5:]);name=node.func.id
            args=[ast.unparse(a) for a in node.args]
            emit(f'call_out=_call({name!r},('+','.join(expr(a) for a in args)+',),tables,xp)  # WRF:'+str(n))
            if name=='var_check':outs=list(range(5,16))
            elif name=='access_lookup_table':outs=[10]
            elif name=='wet_growth_check':outs=[14]
            elif name=='aggregation':outs=[11,18,19,20,21,22,23]
            elif name=='vaporgrow':outs=[32,33,34,35,36,37,38,39,40,41,43,44]
            else:raise ValueError((n,name))
            for j,i in enumerate(outs):assign(args[i],f'call_out[{j}]',n,mask)
            continue
        lhs,rhs=map(str.strip,s.split('=',1));cap=None
        if lhs=='n0rr' and 'lamr**4' in rhs:
            assert previous_source in (('lamr','lamminr'),('lamr','lammaxr')),(n,'Original capped-LAMR assignment must remain adjacent',previous_source)
            cap=previous_source[1]
        assign(lhs,expr(rhs,cap),n,mask)
        previous_source=(lhs,rhs)
    assert not stack,stack
    header=['"""Generated original Jensen '+phase+'; see lane port_levels proof. Not yet qualified."""',
            'from .level_support import *  # noqa: F403','','',f'def {phase}(fields,tables,*,xp=jnp,observer=None):',
            '    v=dict(fields)','    _,_r=_real(v["qv"],xp)','    cat=3;ice1=1;ice2=2;ice3=3',
            '    riming=rain_ice=freeze_qc=splinters=wet_grow=True',
            '    _nint=lambda a: _fortran_nint(a,xp)','    _int=lambda a: xp.asarray(a).astype("int32")',
            '    _max=lambda *a: _maximum(a,xp)','    _min=lambda *a: _minimum(a,xp)',
            '    _expand=lambda a,n: _trailing(a,n,xp)']
    for name,rhs in parameters.items():
        if name not in static:header.append(f'    v[{name!r}]={expr(rhs)}')
    for name in sorted(assigned):
        kind,dims=metadata.get(name,('real',()))
        shape=tuple(3 if d=='cat' else int(d) for d in dims)
        header.append(f'    v.setdefault({name!r},_initial(v["qv"],{kind!r},{shape!r},xp))')
    text='\n'.join(header+out+['    return v',''])
    ast.parse(text)
    (ROOT/f'src/gpuwrf/physics/jensen_ishmael/{phase}.py').write_text(text)
    print(phase,len(statements),'statements',len(text.splitlines()),'lines',len(assigned),'assigned')


if __name__=='__main__':
    generate('processes',684,2727)
    generate('final_checks',2841,2960)
