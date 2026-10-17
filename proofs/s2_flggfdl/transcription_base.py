#!/usr/bin/env python3
"""Offline transcription parser/emitter foundation reused from SSiB lane.
Origin: 715a29d09a085398bb36e68e9ce3f075f7d553b7 proofs/s2_ssib/translate.py.
GFDL adapter overrides shape handling and DO loops; the SSiB 13-trip
assumption MUST NOT be used for radiation.

Original: Offline literal transcription of WRF's self-contained SSiB column routines.

The generated module is ordinary JAX source, not a runtime Fortran interpreter.
No equations, DATA values, or expression association are simplified. Labels use
predication and local backward edges use lax.while_loop; small fixed loops are
unrolled. The oracle, not this translator, independently establishes fidelity.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from pathlib import Path
import re
import hashlib
import json
import ast


def split(text, delim=','):
    level=0; quote=None; start=0; out=[]
    for i,c in enumerate(text):
        if quote:
            if c==quote: quote=None
        elif c in "'\"": quote=c
        elif c=='(': level+=1
        elif c==')': level-=1
        elif c==delim and level==0: out.append(text[start:i].strip()); start=i+1
    out.append(text[start:].strip())
    return out


def lines(raw):
    pending=''; first=0
    for number,line in enumerate(raw.splitlines(),1):
        # All SSiB comments are free form; no quoted exclamation marks.
        line=line.split('!',1)[0].strip().lower()
        if not line: continue
        if not pending: first=number
        pending += ' '+line.strip('&').strip()
        if line.endswith('&'): continue
        for statement in split(pending,';'):
            if statement: yield first,statement
        pending=''
    assert not pending


LEX=re.compile(r"\s*(\d+\.\d*[ed][+-]?\d+|\.\d+[ed][+-]?\d+|\d*\.\d+|\d+\.(?![a-z])|\d+[ed][+-]?\d+|\d+|\.[a-z]+\.|[a-z_]\w*|'[^']*'|\*\*|<=|>=|==|/=|[^\s])")
PREC={'.or.':1,'.and.':2,'.eq.':3,'.ne.':3,'.lt.':3,'.le.':3,'.gt.':3,'.ge.':3,'==':3,'/=':3,'<':3,'<=':3,'>':3,'>=':3,'+':4,'-':4,'*':5,'/':5,'**':7}
BOPS={'.or.':'|','.and.':'&','.eq.':'==','.ne.':'!=','.lt.':'<','.le.':'<=','.gt.':'>','.ge.':'>=','/=':'!='}
INTR={'exp':'jnp.exp','sqrt':'jnp.sqrt','alog':'jnp.log','log':'jnp.log','abs':'jnp.abs','dabs':'jnp.abs','sin':'jnp.sin','cos':'jnp.cos','tan':'jnp.tan','atan':'jnp.arctan','acos':'jnp.arccos','asin':'jnp.arcsin','tanh':'jnp.tanh','float':'_r','real':'_r','dble':'jnp.float64','int':'jnp.int32','ifix':'jnp.int32','sign':'_sign','nint':'_nint','amin1':'_min','amax1':'_max','min':'_min','max':'_max','iabs':'jnp.abs','trim':'_identity'}
STRINGS={'usgs':0,'ssib':1,'modified_igbp_modis_noah':2,'modis':2}
MODULE={}


def array_index(routine,name,argument,axis):
    dimensions=routine.shapes.get(name,())
    if axis<len(dimensions) and ':' in dimensions[axis]:
        lower=dimensions[axis].split(':',1)[0].strip()
        if re.fullmatch(r'[+-]?\d+',lower):
            return argument if int(lower)==0 else f'({argument})-({lower})'
    return f'({argument})-1'

@dataclass
class Routine:
    name:str
    args:list[str]
    types:dict=field(default_factory=dict)
    shapes:dict=field(default_factory=dict)
    data:dict=field(default_factory=dict)
    params:dict=field(default_factory=dict)
    functions:dict=field(default_factory=dict)
    variables:set=field(default_factory=set)
    body:list=field(default_factory=list)

    def kind(self,name):
        if name in self.types: return self.types[name]
        if name in MODULE: return MODULE[name][0]
        return 'i' if name[0] in 'ijklmn' else 'r'

class Expr:
    def __init__(self,text,r):
        self.tokens=LEX.findall(text); self.pos=0; self.r=r
    def pop(self):
        token=self.tokens[self.pos]; self.pos+=1; return token
    def peek(self): return self.tokens[self.pos] if self.pos<len(self.tokens) else ''
    def parse(self,minprec=0):
        token=self.pop()
        if token in ('+','-','.not.'):
            rhs,kind=self.parse(6)
            lhs=f'({"~" if token==".not." else token}{rhs})'
        elif token=='(':
            lhs,kind=self.parse(); assert self.pop()==')'
        elif token in ('.true.','.false.'):
            lhs='jnp.bool_('+str(token=='.true.')+')'; kind='b'
        elif token[0].isdigit() or token[0]=='.':
            kind='d' if 'd' in token else 'r' if any(c in token for c in '.e') else 'i'
            lhs=f'{ {"r":"_r","d":"jnp.float64","i":"jnp.int32"}[kind]}({token.replace("d","e")})'
        elif token.startswith("'"):
            lhs=f'jnp.int32({STRINGS[token[1:-1]]})'; kind='i'
        else:
            name=token
            if self.peek()=='(':
                self.pop(); args=[]
                if self.peek()!=')':
                    while True:
                        args.append(self.parse())
                        if self.peek()!=',': break
                        self.pop()
                assert self.pop()==')'
                if name in self.r.shapes or name in MODULE and MODULE[name][1]:
                    self.r.variables.add(name)
                    cast={'i':'jnp.int32','b':'jnp.bool_','d':'jnp.float64','r':'_r'}[self.r.kind(name)]
                    base=f"v[{name!r}]" if name not in MODULE or name in self.r.shapes else f"{cast}(_TABLES[{name!r}])"
                    lhs=base+'['+', '.join(array_index(self.r,name,a,i) for i,(a,k) in enumerate(args))+']'; kind=self.r.kind(name)
                elif name in self.r.functions:
                    argnames,expr=self.r.functions[name]
                    assert len(argnames)==len(args)
                    body,bkind=expression(expr,self.r)
                    for arg,(a,k) in zip(argnames,args): body=body.replace(f"v[{arg!r}]",f'({a})')
                    lhs=body; kind=bkind
                elif name in INTR:
                    lhs=INTR[name]+'('+', '.join(a for a,k in args)+')'
                    kind='i' if name in ('int','ifix','nint') else 'd' if name=='dble' else 'r' if name in ('real','float') else ('d' if any(k=='d' for a,k in args) else args[0][1])
                else: raise ValueError(('unknown function',self.r.name,name))
            else:
                self.r.variables.add(name)
                if name in self.r.params:
                    cast={'i':'jnp.int32','b':'jnp.bool_','d':'jnp.float64','r':'_r'}[self.r.kind(name)]
                    lhs=f'{cast}({expression(self.r.params[name],self.r)[0]})'
                elif name in MODULE and name not in self.r.args and name not in self.r.types and not MODULE[name][1]:
                    cast={'i':'jnp.int32','b':'jnp.bool_','d':'jnp.float64','r':'_r'}[MODULE[name][0]]
                    lhs=f'{cast}({expression(MODULE[name][2],self.r)[0]})'
                elif name in MODULE and MODULE[name][1] and name not in self.r.types and name not in self.r.shapes:
                    cast={'i':'jnp.int32','b':'jnp.bool_','d':'jnp.float64','r':'_r'}[self.r.kind(name)]
                    lhs=f"{cast}(_TABLES[{name!r}])"
                else: lhs=f"v[{name!r}]"
                kind=self.r.kind(name)
        while self.peek() in PREC and PREC[self.peek()]>=minprec:
            op=self.pop(); rhs,rkind=self.parse(PREC[op]+(0 if op=='**' else 1))
            # Fortran converts INTEGER operands before mixed REAL arithmetic.
            # NumPy strong float32/int32 scalar arithmetic otherwise promotes
            # to float64, hiding a source-precision change until assignment.
            if op!='**' and PREC[op]>3 and ('i' in (kind,rkind)) and any(k in ('r','d') for k in (kind,rkind)):
                cast='jnp.float64' if 'd' in (kind,rkind) else '_r'
                if kind=='i':lhs=f'{cast}({lhs})'
                if rkind=='i':rhs=f'{cast}({rhs})'
            if op=='**' and rkind=='i' and rhs=='jnp.int32(2)':
                # Native REAL**INTEGER(2) is one REAL multiply. NumPy scalar
                # power with an int32 exponent promotes to float64, and JAX's
                # array-exponent route needlessly introduces general power.
                lhs=f'jnp.square({lhs})'
            elif op=='/' and kind==rkind=='i': lhs=f'jnp.int32(jnp.trunc({lhs}.astype(_r) / {rhs}.astype(_r)))'
            else: lhs=f'({lhs} {BOPS.get(op,op)} {rhs})'
            kind='b' if PREC[op]<=3 else 'd' if 'd' in (kind,rkind) else 'r' if 'r' in (kind,rkind) else 'i'
        return lhs,kind


def expression(text,r):
    e=Expr(text.strip(),r); result=e.parse()
    assert e.pos==len(e.tokens),(r.name,text,e.tokens[e.pos:])
    return result


def decl(text,r):
    if text.startswith(('real','integer','character','logical','dimension')):
        kind='i' if text.startswith('integer') else 'b' if text.startswith('logical') else 'd' if re.match(r'real\s*\(\s*kind\s*=\s*k15\s*\)',text) else 'r'
        if text.startswith('character'): return True
        if '::' in text:
            prefix,names=text.split('::',1)
            dim=re.search(r'dimension\s*\(([^)]+)\)',prefix)
            shared=split(dim[1]) if dim else None
        else:
            names=re.sub(r'^(real|integer|logical|dimension)\s*','',text); prefix=text[:len(text)-len(names)]; shared=None
        for item in split(names):
            if '=' in item:
                name,value=item.split('=',1); name=name.strip(); r.params[name]=value.strip(); r.types[name]=kind
                continue
            m=re.fullmatch(r'(\w+)\s*(?:\(([^)]+)\))?',item)
            assert m,(r.name,text,item)
            name=m[1]; shape=split(m[2]) if m[2] else shared
            if not text.startswith('dimension'): r.types[name]=kind
            if shape: r.shapes[name]=shape
        return True
    if text.startswith('parameter'):
        for item in split(text[text.index('(')+1:text.rindex(')')]):
            name,value=item.split('=',1); r.params[name.strip()]=value.strip()
        return True
    if text.startswith('data '):
        rest=text[5:].strip()
        while rest:
            names,rest=rest.split('/',1); vals,rest=rest.split('/',1)
            names=split(names.strip().lstrip(',')); values=[]
            for token in split(vals):
                repeat=re.fullmatch(r'(\d+)\s*\*\s*(.+)',token)
                values.extend([repeat[2]]*int(repeat[1]) if repeat else [token])
            if len(names)==1: r.data[names[0]]=values
            else:
                assert len(names)==len(values),(r.name,names,values)
                r.data.update({n:[v] for n,v in zip(names,values)})
            rest=rest.strip().lstrip(',').strip()
        return True
    if text.startswith(('implicit ','save','external ')): return True
    return False

@dataclass
class Node:
    text:str
    line:int
    label:int=0
    children:list=field(default_factory=list)
    other:list=field(default_factory=list)


def condition(text):
    start=text.index('('); level=0
    for p in range(start,len(text)):
        if text[p]=='(': level+=1
        if text[p]==')':
            level-=1
            if level==0: return text[start+1:p],text[p+1:].strip()
    raise ValueError(text)


def parsebody(items,r):
    def parse_if(node,pos,stoplabel):
        node.children,pos=block(pos,stoplabel)
        if pos<len(items) and items[pos][1].startswith(('else if','elseif')):
            ln2,t=items[pos]; cond,tail=condition(t); assert tail=='then'
            nested=Node('__if__ '+cond,ln2)
            nested,pos=parse_if(nested,pos+1,stoplabel)
            node.other=[nested]
            return node,pos
        if pos<len(items) and items[pos][1]=='else':
            node.other,pos=block(pos+1,stoplabel)
        assert pos<len(items) and items[pos][1] in ('end if','endif'),(r.name,items[pos:] if pos<len(items) else 'EOF',node)
        return node,pos+1
    def block(pos,stoplabel=None):
        out=[]
        while pos<len(items):
            ln,text=items[pos]; label=0
            labeled=re.match(r'^(\d+)\s+(.*)',text)
            if labeled: label=int(labeled[1]); text=labeled[2]
            if text.startswith(('end if','endif','else','end do','enddo')): break
            node=Node(text,ln,label); pos+=1
            if decl(text,r): continue
            fn=re.fullmatch(r'(\w+)\((\w+)\)\s*=\s*(.*)',text)
            if fn and fn[1] not in r.shapes:
                r.functions[fn[1]]=([fn[2]],fn[3]); continue
            if re.match(r'if\s*\(',text) and condition(text)[1]=='then':
                node.text='__if__ '+condition(text)[0]
                node,pos=parse_if(node,pos,stoplabel)
            elif text.startswith('do '):
                m=re.match(r'do\s+(?:(\d+)\s+)?(\w+)\s*=\s*(.*)',text)
                assert m,(r.name,text)
                terminal=int(m[1]) if m[1] else None
                node.text='do '+m[2]+'='+m[3]
                node.children,pos=block(pos,terminal)
                if terminal:
                    if pos<len(items) and re.match(r'^'+str(terminal)+r'\s',items[pos][1]):
                        ln3,t=items[pos]; node.children.append(Node(re.sub(r'^\d+\s+','',t),ln3,terminal)); pos+=1
                    assert any(n.label==terminal for n in descend(node.children)),(r.name,'missing terminal',terminal,pos)
                else:
                    assert items[pos][1] in ('end do','enddo'),(r.name,items[pos]); pos+=1
            out.append(node)
            if stoplabel and label==stoplabel: break
            if stoplabel and any(n.label==stoplabel for n in descend(node.children)): break
        return out,pos
    result,pos=block(0)
    assert pos==len(items),(r.name,pos,items[pos:pos+3])
    return result


def descend(nodes):
    for n in nodes:
        yield n
        yield from descend(n.children); yield from descend(n.other)


def gotos(node):
    return [int(m[1]) for n in descend([node]) if (m:=re.search(r'go\s*to\s*(\d+)',n.text))]

class Emitter:
    def __init__(self,r): self.r=r; self.output=[]; self.counter=0
    def emit(self,text,depth=1): self.output.append('    '*depth+text)
    def fresh(self): self.counter+=1; return f'_p{self.counter}'
    def assign(self,lhs,rhs,mask,depth):
        m=re.fullmatch(r'(\w+)\s*(?:\((.*)\))?',lhs.strip()); assert m,lhs
        name=m[1]
        if name in self.r.params or name in MODULE and name not in self.r.args and name not in self.r.types and name not in self.r.shapes:
            return
        self.r.variables.add(name)
        kind=self.r.kind(name); cast={'i':'jnp.int32','b':'jnp.bool_','r':'_r','d':'jnp.float64'}[kind]
        value=f'{cast}({rhs})'
        if m[2]:
            indices=', '.join(array_index(self.r,name,expression(t,self.r)[0],i) for i,t in enumerate(split(m[2])))
            self.emit(f"v[{name!r}] = v[{name!r}].at[{indices}].set(jnp.where({mask}, {value}, v[{name!r}][{indices}]))",depth)
        elif name in self.r.shapes:
            self.emit(f"v[{name!r}] = jnp.where({mask}, {value}, v[{name!r}])",depth)
        else: self.emit(f"v[{name!r}] = jnp.where({mask}, {value}, v[{name!r}])",depth)
    def body(self,nodes,depth=1,parent='jnp.bool_(True)',ignore=frozenset()):
        # Backward edges form local iteration regions in the original routine.
        index=0
        while index<len(nodes):
            node=nodes[index]
            if node.label and node.label not in ignore:
                ends=[j for j in range(index,len(nodes)) if node.label in gotos(nodes[j])]
                if ends:
                    end=max(ends); fname=self.fresh()
                    self.emit(f'def {fname}(v):',depth)
                    self.emit('v = dict(v)',depth+1)
                    self.emit(f"v['_go'] = jnp.where(v['_go']=={node.label}, 0, v['_go'])",depth+1)
                    self.body(nodes[index:end+1],depth+1,parent,ignore|{node.label})
                    self.emit('return v',depth+1)
                    self.emit(f'v = {fname}(v)',depth)
                    self.emit(f"v = lax.while_loop(lambda v: v['_go']=={node.label}, {fname}, v)",depth)
                    index=end+1; continue
            self.node(node,depth,parent); index+=1
    def node(self,n,depth,parent):
        text=n.text
        if n.label: self.emit(f"v['_go'] = jnp.where(({parent}) & (v['_go']=={n.label}), 0, v['_go'])",depth)
        mask=f"(({parent}) & (v['_go']==0))"
        self.emit(f'# WRF:{n.line} {text}',depth)
        if text in ('continue','return'):
            if text=='return': self.emit(f"v['_go'] = jnp.where({mask}, -1, v['_go'])",depth)
            return
        if text.startswith(('stop','call wrf_error_fatal')):
            # No device-side host callback: make WRF's fatal branch explicitly
            # nonfinite so the standard pre-mask finiteness gate rejects it.
            for name in self.r.args:
                if self.r.kind(name)=='r':
                    self.assign(name,'_r(jnp.nan)',mask,depth)
            self.emit(f"v['_go'] = jnp.where({mask}, -1, v['_go'])",depth)
            return
        if text.startswith(('write','print','call wrf_','format')): return
        if text=='exit':
            self.emit(f"v['_go'] = jnp.where({mask}, -2, v['_go'])",depth); return
        if text.startswith('__if__ '):
            pred=self.fresh(); self.emit(f'{pred} = {mask} & ({expression(text[7:],self.r)[0]})',depth)
            inv=self.fresh(); self.emit(f'{inv} = {mask} & ~({expression(text[7:],self.r)[0]})',depth)
            self.body(n.children,depth,pred); self.body(n.other,depth,inv); return
        if text.startswith('if(') or text.startswith('if ('):
            cond,tail=condition(text); pred=self.fresh()
            self.emit(f'{pred} = {mask} & ({expression(cond,self.r)[0]})',depth)
            self.node(Node(tail,n.line),depth,pred); return
        if text.startswith('do '):
            name,lim=text[3:].split('=',1); lim=split(lim); step=lim[2] if len(lim)>2 else '1'
            args=[expression(x,self.r)[0] for x in (lim[0],lim[1],step)]
            # All column-profile DO loops have a static bound <=13. ND and NK
            # are runtime snow layer counts, bounded by the four-slot schema.
            def static(x):
                x=x.strip()
                if re.fullmatch(r'[+-]?\d+',x): return int(x)
                if x in MODULE and not MODULE[x][1]: return int(float(MODULE[x][2]))
                return None
            a,b,c=map(static,(lim[0],lim[1],step)); assert c is not None
            maxiter=len(range(a,b+(1 if c>0 else -1),c)) if a is not None and b is not None else 13
            pred=self.fresh(); self.emit(f'{pred} = {mask}',depth)
            iv=self.fresh(); self.emit(f'for {iv} in range({maxiter}):',depth)
            value=f'({args[0]} + {iv} * {args[2]})'
            itmask=f"({pred} & (v['_go']==0) & ({value} {'<=' if c>0 else '>='} {args[1]}))"
            iterpred=self.fresh(); self.emit(f'{iterpred} = {itmask}',depth+1)
            self.assign(name,value,iterpred,depth+1)
            self.body(n.children,depth+1,iterpred)
            self.assign(name,f'({args[1]} + {args[2]})',f"({pred} & (v['_go']==0))",depth)
            self.emit("v['_go'] = jnp.where(v['_go']==-2, 0, v['_go'])",depth)
            return
        m=re.fullmatch(r'go\s*to\s*(\d+)',text)
        if m:
            self.emit(f"v['_go'] = jnp.where({mask}, {m[1]}, v['_go'])",depth); return
        if text.startswith('call '):
            m=re.fullmatch(r'call\s+(\w+)\s*\((.*)\)',text); assert m,text
            name=m[1]; actual=split(m[2]); vals=[expression(x,self.r)[0] for x in actual]
            temp=self.fresh()
            self.emit(f"{temp} = lax.cond({mask}, lambda _a: _{name}(*_a), lambda _a: _a, ({', '.join(vals)},))",depth)
            # Every nonconstant argument is returned, preserving INOUT aliasing.
            for i,x in enumerate(actual):
                if re.fullmatch(r'[a-z_]\w*(?:\(.*\))?',x):
                    self.assign(x,f'{temp}[{i}]',mask,depth)
            return
        if '=' in text:
            lhs,rhs=text.split('=',1)
            if lhs.strip()=='message': return
            self.assign(lhs,expression(rhs,self.r)[0],mask,depth); return
        raise ValueError((self.r.name,n.line,text))
    def render(self):
        self.body(self.r.body)
        # Statement functions introduce dummy names while parsing; they need no
        # runtime state unless also used elsewhere (zero scratch is harmless).
        initial=[]
        for name in sorted(self.r.variables | set(self.r.args)):
            if name in MODULE and name not in self.r.shapes and name not in self.r.args and name not in self.r.types: continue
            if name in self.r.params: continue
            if name in self.r.args: continue
            kind=self.r.kind(name); cast={'i':'jnp.int32','b':'jnp.bool_','r':'_r','d':'jnp.float64'}[kind]
            dims=self.r.shapes.get(name)
            if name in self.r.data:
                data=self.r.data[name]
                values=[f'{cast}({expression(x,self.r)[0]})' for x in data]
                value='jnp.stack(['+', '.join(values)+'])'
                if dims: value+=f'.reshape({self.shape(dims)!r}, order="F")'
                else: assert len(data)==1; value=values[0]
            elif dims: value=f'jnp.zeros({self.shape(dims)!r}, dtype={cast})'
            else: value=f'{cast}(0)'
            initial.append(f'    v[{name!r}] = {value}')
        # DATA can initialize individual array elements (INTERCS/INTERC PCOEFS),
        # not just whole arrays. Keep each literal assignment in Fortran order.
        for target,values in self.r.data.items():
            item=re.fullmatch(r'(\w+)\((.*)\)',target)
            if item:
                assert len(values)==1
                name=item[1];index=tuple(int(x)-1 for x in split(item[2]))
                cast='jnp.int32' if self.r.kind(name)=='i' else '_r'
                value=expression(values[0],self.r)[0]
                initial.append(f"    v[{name!r}] = v[{name!r}].at[{index!r}].set({cast}({value}))")
        # UPDAT1's native multilayer branch bypasses SNOWW(2)'s assignment.
        # Manager-approved zero-init reference policy; no hidden cross-column
        # local stack memory is carried into a GPU column.
        header=[f'def _{self.r.name}('+', '.join(self.r.args)+'):',
                '    _r = jnp.float32 if jnp.asarray('+next((a for a in self.r.args if self.r.kind(a)=='r'),'0.0')+').dtype == jnp.dtype("float32") else jnp.float64',
                '    v = {'+', '.join(f'{a!r}: {a}' for a in self.r.args)+'}',
                "    v['_go'] = jnp.int32(0)"]
        ret='    return ('+', '.join(f'v[{a!r}]' for a in self.r.args)+',)'
        return '\n'.join(header+initial+self.output+[ret])
    def shape(self,dims):
        out=[]
        for x in dims:
            if x.isdigit(): out.append(int(x))
            elif x in self.r.params: out.append(int(float(self.r.params[x])))
            elif x in MODULE: out.append(int(float(MODULE[x][2])))
            else: raise ValueError((self.r.name,'dynamic array shape',x))
        return tuple(out)

