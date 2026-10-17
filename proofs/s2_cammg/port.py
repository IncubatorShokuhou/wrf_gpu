"""Offline, source-line-preserving translation of the frozen WRF CAM-MG routines.

Emits ordinary Python; no parser/interpreter is used by forecasts. Scope is the
nonchem CAM5.1 WRF source identified in the oracle manifest. Unsupported syntax
is a generation error rather than silently omitted physics.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
from pathlib import Path
import re

GLOBAL_ARRAYS={'qmin','pcf','estbl'}


def split(text, delimiter=','):
    result=[];start=0;depth=0;quote=None
    for i,c in enumerate(text):
        if quote:
            if c==quote:quote=None
        elif c in "'\"":quote=c
        elif c=='(':depth+=1
        elif c==')':depth-=1
        elif c==delimiter and depth==0:
            result.append(text[start:i].strip());start=i+1
    result.append(text[start:].strip())
    return result


def lines(text):
    pending='';start=0
    for n,raw in enumerate(text.splitlines(),1):
        # All quotes in these modules are single-line Fortran character literals.
        quote=None;end=len(raw)
        for i,c in enumerate(raw):
            if quote:
                if c==quote:quote=None
            elif c in "'\"":quote=c
            elif c=='!':end=i;break
        raw=raw[:end].strip().lower()
        if not raw or raw.startswith('#'):continue
        if not pending:start=n
        pending+=' '+raw.lstrip('&').rstrip('&').strip()
        if raw.endswith('&'):continue
        for stmt in split(pending.strip(),';'):
            if stmt:yield start,stmt
        pending=''
    assert not pending


@dataclass
class Variable:
    kind:str
    dims:list[str]=field(default_factory=list)
    intent:str=''
    initial:str|None=None
    parameter:bool=False
    allocatable:bool=False


def declaration(stmt,variables):
    legacy=re.fullmatch(r'parameter\s*\((.*)\)',stmt)
    if legacy:
        for item in split(legacy[1]):
            name,value=item.split('=',1);name=name.strip()
            assert name in variables,('undeclared PARAMETER',name)
            variables[name].parameter=True;variables[name].initial=value.strip()
        return True
    match=re.match(r'^(real(?:\s*\([^)]*\)|\*\d+)?|integer(?:\s*\([^)]*\))?|logical|character(?:\s*\([^)]*\)|\*\d+)?)\s*(.*)$',stmt)
    if not match:return False
    typ,rest=match.groups()
    if 'function' in rest:return False
    kind='f64' if ('r8' in typ or '*8' in typ) else 'f32' if typ.startswith('real') else 'i32' if typ.startswith('integer') else 'bool' if typ=='logical' else 'str'
    attrs=''
    if '::' in rest:attrs,rest=rest.split('::',1)
    dimmatch=re.search(r'dimension\s*\(([^)]*)\)',attrs)
    dims=split(dimmatch[1]) if dimmatch else []
    im=re.search(r'intent\s*\((\w+)\)',attrs)
    for var in split(rest):
        m=re.fullmatch(r'(\w+)\s*(?:\((.*?)\))?\s*(?:=\s*(.*))?',var)
        if not m:raise ValueError(('declaration',stmt,var))
        name,shape,initial=m.groups()
        variables[name]=Variable(kind,split(shape) if shape else dims,im[1] if im else '',initial,'parameter' in attrs,'allocatable' in attrs)
    return True


@dataclass
class Procedure:
    name:str
    args:list[str]
    variables:dict[str,Variable]
    statements:list[tuple[int,str]]
    result:str|None=None
    module:str=''


def procedures(text):
    logical=list(lines(text));out={};i=0
    module=next((s.split()[1] for _,s in logical if re.fullmatch(r'module\s+\w+',s)),'')
    while i<len(logical):
        n,s=logical[i]
        match=re.match(r'(?:real\(r8\)\s+|integer\s+)?(subroutine|function)\s+(\w+)\s*\((.*)\)(?:\s+result\((\w+)\))?$',s)
        if not match:i+=1;continue
        typ,name,args,result=match.groups();args=split(args);vars={};stmts=[];i+=1;rk8=False
        while i<len(logical):
            n,s=logical[i];i+=1
            if re.match(r'end\s*(subroutine|function)',s):break
            if re.match(r'integer.*\brk\s*=\s*r8\b',s):rk8=True
            if rk8:s=s.replace('(rk)','(r8)').replace('_rk','_r8')
            if name=='dropmixnuc' and s.startswith("call outfld('ndropcol'"):
                # Subsequent CCN/history computations have no caller-visible
                # scientific side effects. Keep original physical updates.
                while i<len(logical) and not logical[i][1].startswith('end subroutine'):i+=1
                i+=1;break
            if name=='dropmixnuc':
                if '=> qqcw_inout' in s or s.startswith('type(qqcw_type)'):continue
                s=re.sub(r'qqcw\((\w+)\)%fldcw\(([^)]*)\)',r'qqcw_inout(\2,\1)',s)
                if s.startswith('call loadaer('):s=re.sub(r'\bqqcw\b','qqcw_inout',s)
            if name=='loadaer':
                if s.startswith('type(qqcw_type)'):
                    s='real(r8),intent(in)::qqcw(pcols,pver,pcnst)'
                s=re.sub(r'qqcw\((\w+)\)%fldcw\(([^)]*)\)',r'qqcw(\2,\1)',s)
            s=re.sub(r'^\w+\s*:\s*(?=do\b)','',s)
            if s.startswith('allocate('):
                for item in split(s[s.index('(')+1:s.rindex(')')]):
                    allocated=re.fullmatch(r'(\w+)\((.*)\)',item)
                    if allocated and allocated[1] in vars:vars[allocated[1]].dims=split(allocated[2])
                continue
            if s.startswith('deallocate('):continue
            if declaration(s,vars):continue
            if s.startswith(('use ','implicit ','save','private','public','external','intrinsic')):continue
            if s.startswith('dimension '):
                for v in split(s[10:]):
                    m=re.match(r'(\w+)\((.*)\)',v);vars[m[1]].dims=split(m[2])
                continue
            stmts.append((n,s))
        out[name]=Procedure(name,args,vars,stmts,(result or name) if typ=='function' else None,module)
    return out


def expression(text):
    text=text.strip()
    text=re.sub(r'\b(as|lambda|del)\b',r'wrf_\1',text)
    for a,b in (('.and.',' and '),('.or.',' or '),('.not.',' not '),('.ge.','>='),('.gt.','>'),('.le.','<='),('.lt.','<'),('.eq.','=='),('.ne.','!='),('/=','!='),('.true.','True'),('.false.','False')):
        text=text.replace(a,b)
    text=re.sub(r'(?<![\w.])((?:\d+\.\d*|\.\d+|\d+)(?:[ed][+-]?\d+)?)[_]r8\b',r'_d(\1)',text)
    text=re.sub(r'\b((?:\d+\.\d*|\.\d+|\d+)(?:[e][+-]?\d+)?)[_]r4\b',r'\1',text)
    text=re.sub(r'(?<=\d)d([+-]?\d+)',r'e\1',text)
    # A tuple-shaped array argument is represented by a Python slice. Wrap the
    # array call as subscript before AST parsing, then retain Fortran indexing.
    text=re.sub(r'\b(\w+)\(([^()]*(?::)[^()]*)\)',r'\1[\2]',text)
    return ast.parse(text.strip(),mode='eval').body


INTRINSICS={'abs':'abs','sqrt':'sqrt','exp':'exp','log':'log','log10':'log10',
            'sin':'sin','cos':'cos','acos':'arccos','atan':'arctan','tanh':'tanh',
            'floor':'floor','aint':'trunc','mod':'mod','sum':'sum','maxval':'max','minval':'min',
            'any':'any','all':'all'}


class Expression(ast.NodeTransformer):
    def __init__(self,arrays,variables=None):self.arrays=arrays;self.variables=variables or {}

    def visit_BinOp(self,node):
        if isinstance(node.op,ast.Pow):
            if isinstance(node.right,ast.Constant) and isinstance(node.right.value,int):
                return ast.Call(func=ast.Name(id='_powi',ctx=ast.Load()),args=[self.visit(node.left),node.right],keywords=[])
            return ast.Call(func=ast.Name(id='_pow',ctx=ast.Load()),args=[self.visit(node.left),self.visit(node.right),ast.Name(id='xp',ctx=ast.Load())],keywords=[])
        return self.generic_visit(node)

    def visit_Name(self,node):
        name=node.id.removeprefix('wrf_')
        return ast.Subscript(value=ast.Name(id='v',ctx=ast.Load()),slice=ast.Constant(name),ctx=ast.Load())

    def visit_Constant(self,node):
        if isinstance(node.value,float):return ast.Call(func=ast.Name(id='_r',ctx=ast.Load()),args=[node],keywords=[])
        return node

    def visit_BoolOp(self,node):
        fun='logical_and' if isinstance(node.op,ast.And) else 'logical_or'
        values=[self.visit(x) for x in node.values];r=values[0]
        for x in values[1:]:r=ast.Call(func=ast.Attribute(value=ast.Name(id='xp',ctx=ast.Load()),attr=fun,ctx=ast.Load()),args=[r,x],keywords=[])
        return r

    def visit_UnaryOp(self,node):
        if isinstance(node.op,ast.Not):
            return ast.Call(func=ast.Attribute(value=ast.Name(id='xp',ctx=ast.Load()),attr='logical_not',ctx=ast.Load()),args=[self.visit(node.operand)],keywords=[])
        if isinstance(node.op,ast.USub) and isinstance(node.operand,ast.Constant):return self.visit(ast.Constant(-node.operand.value))
        if isinstance(node.op,ast.USub):
            return ast.Call(func=ast.Name(id='_neg',ctx=ast.Load()),args=[self.visit(node.operand)],keywords=[])
        return self.generic_visit(node)

    def visit_Call(self,node):
        name=node.func.id
        if name in self.arrays:return self.array(name,node.args)
        if name=='_d':
            # A KIND-qualified literal is rounded once to DOUBLE, not REAL4 first.
            return ast.Call(func=ast.Name(id='_d',ctx=ast.Load()),args=node.args,keywords=[])
        args=[self.visit(a) for a in node.args]
        if name in ('max','min','max0','min0'):
            fn=ast.Attribute(value=ast.Name(id='xp',ctx=ast.Load()),attr='maximum' if name.startswith('max') else 'minimum',ctx=ast.Load())
            r=args[0]
            for a in args[1:]:r=ast.Call(func=fn,args=[r,a],keywords=[])
            return r
        if name in ('real','float','sngl','dble','int'):
            fn='_int' if name=='int' else '_d' if name=='dble' or len(node.args)>1 else '_r'
            return ast.Call(func=ast.Name(id=fn,ctx=ast.Load()),args=args[:1],keywords=[])
        if name in INTRINSICS:
            return ast.Call(func=ast.Attribute(value=ast.Name(id='xp',ctx=ast.Load()),attr=INTRINSICS[name],ctx=ast.Load()),args=args,keywords=[])
        return ast.Call(func=ast.Name(id=name,ctx=ast.Load()),args=args,keywords=[ast.keyword(arg='xp',value=ast.Name(id='xp',ctx=ast.Load())),ast.keyword(arg='_globals',value=ast.Name(id='_globals',ctx=ast.Load()))])

    def array(self,name,args):
        indices=[]
        for axis,arg in enumerate(args):
            dims=self.variables.get(name,Variable('f64')).dims
            lower=(dims[axis].split(':')[0] or '1') if axis<len(dims) and ':' in dims[axis] else '1'
            def offset(node):
                value=self.visit(node)
                if lower!='1':
                    value=ast.BinOp(left=ast.BinOp(left=value,op=ast.Add(),right=ast.Constant(1)),op=ast.Sub(),right=Expression(self.arrays,self.variables).visit(expression(lower)))
                return value
            if isinstance(arg,ast.Slice):
                begin=offset(arg.lower) if arg.lower else ast.Constant(1)
                upper=offset(arg.upper) if arg.upper else ast.Constant(None)
                step=self.visit(arg.step) if arg.step else ast.Constant(None)
                indices.append(ast.Call(func=ast.Name(id='_slice',ctx=ast.Load()),args=[begin,upper,step],keywords=[]))
            else:indices.append(offset(arg))
        return ast.Call(func=ast.Name(id='_get',ctx=ast.Load()),args=[self.visit_Name(ast.Name(id=name,ctx=ast.Load())),ast.Tuple(elts=indices,ctx=ast.Load())],keywords=[])

    def visit_Subscript(self,node):
        assert isinstance(node.value,ast.Name)
        args=node.slice.elts if isinstance(node.slice,ast.Tuple) else [node.slice]
        return self.array(node.value.id,args)


def expr(text,variables):
    arrays={k for k,v in variables.items() if v.dims}|GLOBAL_ARRAYS
    return ast.unparse(ast.fix_missing_locations(Expression(arrays,variables).visit(expression(text))))


@dataclass
class Node:
    kind:str
    text:str
    line:int
    body:list=field(default_factory=list)
    other:list=field(default_factory=list)


def parse(stmts):
    root=[];current=root;stack=[]
    for n,s in stmts:
        label=re.match(r'^(\d+)\s+(.*)',s)
        if label:
            current.append(Node('label',label[1],n));s=label[2]
            if s=='continue':continue
        if s.startswith(('enddo','end do')):
            kind,parent,node=stack.pop();assert kind=='do',(n,s,kind);current=parent
        elif s.startswith(('endwhere','end where')):
            kind,parent,node=stack.pop();assert kind=='where';current=parent
        elif s.startswith(('endif','end if')):
            while True:
                kind,parent,node=stack.pop();assert kind=='if',(n,s,kind)
                current=parent
                if not getattr(node,'elif_node',False):break
        elif re.match(r'else\s*if|elseif',s):
            m=re.fullmatch(r'(?:else\s*if|elseif)\s*\((.*)\)\s*then',s);assert m,(n,s)
            kind,parent,node=stack[-1];assert kind=='if'
            node.other=[Node('if',m[1],n)];new=node.other[0];new.elif_node=True
            stack.append(('if',node.other,new));current=new.body
        elif s=='else':
            kind,parent,node=stack[-1];assert kind=='if';current=node.other
        elif re.match(r'if\s*\(',s):
            # Match the closing parenthesis of the actual IF condition.
            start=s.index('(');depth=1;i=start+1
            while depth:depth+=(s[i]=='(')-(s[i]==')');i+=1
            cond=s[start+1:i-1];tail=s[i:].strip();node=Node('if',cond,n);current.append(node)
            if tail=='then':stack.append(('if',current,node));current=node.body
            else:node.body=[Node('statement',tail,n)]
        elif re.match(r'do\s',s):
            node=Node('do',s[2:].strip(),n);current.append(node);stack.append(('do',current,node));current=node.body
        elif s.startswith('where '):
            m=re.fullmatch(r'where\s*\((.*)\)',s);assert m,s
            node=Node('where',m[1],n);current.append(node);stack.append(('where',current,node));current=node.body
        else:current.append(Node('statement',s,n))
    assert not stack,stack
    def structure(nodes):
        for node in nodes:
            structure(node.body);structure(node.other)
        while True:
            found=False
            for start,node in enumerate(nodes):
                if node.kind!='label':continue
                target=node.text
                for end in range(start+1,len(nodes)):
                    if any(n.kind=='statement' and re.fullmatch(r'go\s*to\s*'+target+r'|goto\s*'+target,n.text) for n in walk([nodes[end]])):
                        # Only a real backward edge is structured here. The
                        # source trial is re-evaluated with unchanged outer n.
                        nodes[start:end+1]=[Node('retry',target,node.line,nodes[start+1:end+1])]
                        found=True;break
                if found:break
            if not found:break
    structure(root)
    if root and root[-1].kind=='statement' and root[-1].text=='return':root.pop()
    return root


class Emitter:
    def __init__(self,procedure,allprocs):
        self.p=procedure;self.all=allprocs;self.code=[];self.indent=1;self.count=0

    def emit(self,line):self.code.append('    '*self.indent+line)
    def ex(self,text):return expr(text,self.p.variables)

    def has_fatal(self,p=None,seen=()):
        p=p or self.p
        if p.name in seen:return False
        for _,s in p.statements:
            for name in re.findall(r'\bcall\s+(\w+)',s):
                if name in ('wrf_error_fatal','endrun'):return True
                if name in self.all and self.has_fatal(self.all[name],(*seen,p.name)):return True
        return False

    def is_output(self,p,name,seen=()):
        spec=p.variables.get(name,Variable('f64'))
        if spec.intent:return spec.intent!='in'
        if (p.name,name) in seen:return False
        seen=(*seen,(p.name,name))
        for node in walk(parse(p.statements)):
            if node.kind!='statement':continue
            s=node.text
            if re.match(r'^'+re.escape(name)+r'\s*(?:\(.*\))?\s*=(?!=)',s):return True
            if s.startswith('call '):
                callee,args=self.call(s)
                if callee=='cnst_get_ind' and re.match(r'^'+name+r'\b',args[1]):return True
                if callee in self.all:
                    child=self.all[callee]
                    for formal,actual in zip(child.args,args,strict=True):
                        if re.match(r'^'+re.escape(name)+r'\b',actual) and self.is_output(child,formal,seen):return True
        return False

    def lhs(self,text):
        m=re.fullmatch(r'(\w+)\s*(?:\((.*)\))?',text.strip());assert m,text
        name,idx=m.groups()
        if idx is None:return name,'None'
        node=expression(f'{name}({idx})')
        args=node.slice.elts if isinstance(node,ast.Subscript) and isinstance(node.slice,ast.Tuple) else [node.slice] if isinstance(node,ast.Subscript) else node.args
        transform=Expression({k for k,v in self.p.variables.items() if v.dims}|GLOBAL_ARRAYS,self.p.variables)
        indices=[]
        for axis,arg in enumerate(args):
            dims=self.p.variables.get(name,Variable('f64')).dims
            bound=(dims[axis].split(':')[0] or '1') if axis<len(dims) and ':' in dims[axis] else '1'
            def exindex(node):
                value=ast.unparse(ast.fix_missing_locations(transform.visit(node)))
                return f'({value})+1-({self.ex(bound)})' if bound!='1' else value
            if isinstance(arg,ast.Slice):
                lo=exindex(arg.lower) if arg.lower else 'None';hi=exindex(arg.upper) if arg.upper else 'None'
                indices.append(f'_slice({lo},{hi},{self.ex(ast.unparse(arg.step)) if arg.step else "None"})')
            else:indices.append(exindex(arg))
        return name,'('+','.join(indices)+',)'

    def assign(self,lhs,rhs,line):
        name,idx=self.lhs(lhs)
        self.emit(f'_put(v,{name!r},{idx},{rhs})  # WRF:{line}')

    def writes(self,nodes):
        names=set()
        for node in nodes:
            if node.kind in ('if','do','where','retry'):
                names|=self.writes(node.body)|self.writes(node.other)
                if node.kind=='do':names.add(node.text.split('=')[0].strip())
            elif node.kind=='label':names.add('_jump')
            else:
                s=node.text
                if re.match(r'go\s*to\b|goto\b',s) or s=='return':names.add('_jump')
                elif s.startswith('call '):
                    name,args=self.call(s)
                    if name in ('wrf_error_fatal','endrun'):names.add('_fatal');continue
                    if name=='cnst_get_ind':names.add(self.lhs(args[1])[0]);continue
                    if name in self.all:
                        p=self.all[name]
                        assert len(p.args)==len(args),(self.p.name,node.line,name,p.args,args)
                        if self.has_fatal(p):names.add('_fatal')
                        for a,b in zip(p.args,args,strict=True):
                            if self.is_output(p,a):names.add(self.lhs(b)[0])
                    elif name not in ('outfld','wrf_message','wrf_debug'):raise ValueError(('unresolved call',name))
                elif '=' in s and not s.startswith(('write','data ','allocate','deallocate')):names.add(self.lhs(s.split('=',1)[0])[0])
        return {n for n in names if self.p.variables.get(n,Variable('f64')).kind!='str'}

    @staticmethod
    def call(s):
        m=re.fullmatch(r'call\s+(\w+)(?:\s*\((.*)\))?',s);assert m,s
        args=split(m[2]) if m[2] is not None else []
        if m[1]=='explmix' and len(args)==12:
            assert args[-1]=='.false.',args
            args.append('0._r8') # Original optional qactold is unread for ACTIVATED species.
        if m[1]=='explmix':
            for i in (0,6,12):
                first=re.fullmatch(r'(\w+)\(1,(.*)\)',args[i])
                if first:args[i]=f'{first[1]}(1:pver,{first[2]})'
        return m[1],args

    def block(self,nodes,guard_jump=False):
        for node in nodes:
            if guard_jump and node.kind!='label':
                self.branch("v['_jump'] == 0",[node],[],node.line,False)
                continue
            if node.kind=='if':self.branch(self.ex(node.text),node.body,node.other,node.line,True)
            elif node.kind=='where':
                self.count+=1;c=self.count;self.emit(f'_where{c}={self.ex(node.text)}')
                for a in node.body:
                    assert a.kind=='statement' and '=' in a.text
                    lhs,rhs=a.text.split('=',1)
                    self.assign(lhs,f'xp.where(_where{c},{self.ex(rhs)},{self.ex(lhs)})',a.line)
            elif node.kind=='retry':
                names=tuple(sorted(self.writes(node.body)|{'_jump'}));self.count+=1;c=self.count
                self.emit(f'_outer{c}=dict(v)')
                self.emit(f'def _retrybody{c}(_carry):');self.indent+=1
                self.emit(f'v=dict(_outer{c});v.update(zip({names!r},_carry))')
                self.block(node.body,True)
                self.emit(f'return tuple(v[_n] for _n in {names!r})');self.indent-=1
                self.emit(f'_result=_retry({node.text},_retrybody{c},tuple(v[_n] for _n in {names!r}),xp,{names.index("_jump")})')
                self.emit(f'v.update(zip({names!r},_result))')
            elif node.kind=='do':
                m=re.fullmatch(r'(\w+)\s*=\s*(.*)',node.text);assert m,('unsupported do',node)
                name,bounds=m.groups();bounds=split(bounds);assert len(bounds) in (2,3)
                start,end=self.ex(bounds[0]),self.ex(bounds[1]);step=self.ex(bounds[2]) if len(bounds)==3 else '1'
                names=tuple(sorted(self.writes(node.body)|{name,'_jump'}));self.count+=1;c=self.count
                self.emit(f'_outer{c}=dict(v)')
                self.emit(f'def _body{c}(_k,_carry):');self.indent+=1
                self.emit(f'v=dict(_outer{c});v.update(zip({names!r},_carry));v[{name!r}]=_int(_k)')
                self.block(node.body,any(n.kind=='statement' and (re.match(r'go\s*to\b|goto\b',n.text) or n.text=='return') for n in walk(node.body)))
                self.emit(f'return tuple(v[_n] for _n in {names!r})');self.indent-=1
                self.emit(f'_result=_loop({start},{end},{step},_body{c},tuple(v[_n] for _n in {names!r}),xp,jump_index={names.index("_jump")})')
                self.emit(f'v.update(zip({names!r},_result))')
            elif node.kind=='label':self.emit(f"v['_jump']=xp.where(v['_jump']=={node.text},_int(0),v['_jump'])")
            else:self.statement(node)

    def branch(self,cond,yes,no,line,guard):
        names=tuple(sorted(self.writes(yes)|self.writes(no)));self.count+=1;c=self.count
        if not names:return
        self.emit(f'_outer{c}=dict(v)')
        for tag,body in [('yes',yes),('no',no)]:
            self.emit(f'def _{tag}{c}(_carry):');self.indent+=1
            self.emit(f'v=dict(_outer{c});v.update(zip({names!r},_carry))')
            self.block(body,False)
            self.emit(f'return tuple(v[_n] for _n in {names!r})');self.indent-=1
        self.emit(f'_result=_cond({cond},_yes{c},_no{c},tuple(v[_n] for _n in {names!r}),xp)  # WRF:{line}')
        self.emit(f'v.update(zip({names!r},_result))')

    def statement(self,node):
        s=node.text;n=node.line
        if not s or s=='continue':return
        if s=='return':self.emit("v['_jump']=_int(-1)");return
        if s.startswith(('write','print','data ','format(')):return
        if re.match(r'go\s*to\b|goto\b',s):
            label=re.findall(r'\d+',s);assert len(label)==1,s
            self.emit(f"v['_jump']=_int({label[0]})  # WRF:{n}");return
        if s.startswith('call '):
            name,args=self.call(s)
            if name in ('outfld','wrf_message','wrf_debug'):return
            if name in ('wrf_error_fatal','endrun'):
                self.emit(f"v['_fatal']=v['_fatal']+_int(1)  # Original WRF fatal event:{n}; caller must refuse")
                return
            if name=='cnst_get_ind':
                index={'q':1,'cldliq':2,'cldice':3,'numliq':4,'numice':5}[ast.literal_eval(args[0]).lower()]
                self.assign(args[1],f'_int({index})',n);return
            if name=='qneg3':args[0]="''"  # Only the unused diagnostic subroutine label.
            if name=='qneg3':
                # WRF sequence-associates the full constituent slab at its
                # first scalar element. qneg3 has only the qmin floor as a
                # scientific side effect; all remaining writes are logging.
                assert len(args)==9 and args[5]==args[6],args
                m=re.fullmatch(r'(\w+)\(1,1,(\w+)\)',args[8].replace(' ',''));assert m,args[8]
                lhs=f'{m[1]}(:,:,{m[2]})'
                self.assign(lhs,f'xp.maximum({self.ex(lhs)},{self.ex(args[7])})',n)
                return
            assert name in self.all,('missing callee',name,n)
            p=self.all[name];assert len(args)==len(p.args),(name,len(args),len(p.args))
            output=[i for i,a in enumerate(p.args) if self.is_output(p,a)]
            self.emit(f'_call={name}('+','.join(self.ex(a) for a in args)+',_globals=_globals,xp=xp)')
            for j,i in enumerate(output):self.assign(args[i],f'_call[{j}]',n)
            if self.has_fatal(p):self.emit(f"v['_fatal']=v['_fatal']+_call[{len(output)}]")
            return
        if '=' in s:
            lhs,rhs=s.split('=',1)
            name,_=self.lhs(lhs)
            if self.p.variables.get(name,Variable('f64')).kind=='str':
                if re.fullmatch(r"'[^']*'",rhs.strip()):self.emit(f'v[{name!r}]={rhs.strip()}')
                return  # Diagnostic tendency labels never enter scientific arrays.
            self.assign(lhs,self.ex(rhs),n);return
        raise ValueError(('unhandled executable',self.p.name,n,s))

    def render(self):
        p=self.p
        used=set(re.findall(r'\b[a-z_]\w*\b',' '.join(s for _,s in p.statements)))
        while True:
            more={n for k,var in p.variables.items() if k in used for d in var.dims for n in re.findall(r'\b[a-z_]\w*\b',d)}
            if more<=used:break
            used|=more
        self.emit('v=dict(_globals or {})')
        self.emit(f"if '__modules' in v:v.update(v['__modules'].get({p.module!r},{{}}))")
        for a in p.args:self.emit(f'v[{a!r}]={a}')
        self.emit("v['_jump']=_int(0)")
        if self.has_fatal():self.emit("v['_fatal']=_int(0)")
        for name,var in p.variables.items():
            if name in p.args:continue
            if name not in used:continue
            if var.kind=='str':
                self.emit(f'v[{name!r}]={var.initial or repr("")}');continue
            if var.initial is not None:
                initial=var.initial.strip()
                if initial.startswith('(/') and initial.endswith('/)'):
                    values=split(initial[2:-2])
                    self.emit(f"v[{name!r}]=_literal_array(["+','.join(self.ex(x) for x in values)+f"],{var.kind!r},xp)")
                elif var.parameter and var.kind in ('i32','bool'):
                    self.emit(f"v[{name!r}]=_static({self.ex(var.initial)},{var.kind!r})")
                else:self.emit(f"v[{name!r}]=_cast({self.ex(var.initial)},{var.kind!r},xp)")
            else:
                shape='('+','.join(f'({self.ex(x.split(":")[1])})-({self.ex(x.split(":")[0])})+1' if ':' in x else self.ex(x) for x in var.dims)+(',' if var.dims else '')+')'
                self.emit(f"v[{name!r}]=_zeros({shape},{var.kind!r},xp)")
        nodes=parse(p.statements)
        self.block(nodes,any(n.kind=='statement' and re.match(r'go\s*to\b|goto\b',n.text) for n in walk(nodes))
                   or any(n.kind=='statement' and n.text=='return' for n in walk(nodes)))
        outputs=[a for a in p.args if self.is_output(p,a)]
        if self.has_fatal():outputs.append('_fatal')
        self.emit(f'return v[{p.result!r}]' if p.result else 'return ('+','.join(f'v[{a!r}]' for a in outputs)+',)')
        return f'def {p.name}('+','.join(p.args)+',*,_globals=None,xp=jnp):\n'+'\n'.join(self.code)+'\n'


def walk(nodes):
    for n in nodes:
        yield n;yield from walk(n.body);yield from walk(n.other)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('sources',nargs='+',type=Path)
    ap.add_argument('--routines',required=True);ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args();allprocs={}
    for source in args.sources:
        text=source.read_text();allprocs.update(procedures(text));variables={}
        for _,s in lines(text):
            if s=='contains':break
            declaration(s,variables)
        GLOBAL_ARRAYS.update(k for k,v in variables.items() if v.dims)
    selected=args.routines.split(',')
    code='''"""Generated offline from pristine WRF CAM-MG; see proofs/s2_cammg/port.py.
UNQUALIFIED until the original-caller stage gates pass; not admitted to dispatch.
"""\nimport jax.numpy as jnp\nfrom ._support import _r,_d,_int,_get,_put,_slice,_loop,_cond,_cast,_dtype\n\n'''
    code=code.replace('_cast,_dtype','_cast,_dtype,_pow,_powi,_retry,_zeros,_literal_array,_static,_neg')
    external={n for p in selected for _,s in allprocs[p].statements
              for n in ['vqsatd_water','polysvp','gamma','cal_cldfra_mp_1d','nucleati','erf','erfc','estblf'] if n+'(' in s.replace(' ','')}
    external-=set(selected)
    for name in sorted(external):
        module='gamma' if name=='gamma' else 'ice_activation' if name=='nucleati' else 'error_function' if name in ('erf','erfc') else 'saturation_table' if name=='estblf' else 'cloud'
        code+=f'from .{module} import {name}\n'
    for name in selected:code+=Emitter(allprocs[name],allprocs).render()+'\n'
    from dataclasses import asdict
    code+='ARGUMENTS='+repr({n:allprocs[n].args for n in selected})+'\n'
    code+='OUTPUTS='+repr({n:[a for a in allprocs[n].args if Emitter(allprocs[n],allprocs).is_output(allprocs[n],a)]+(['_fatal'] if Emitter(allprocs[n],allprocs).has_fatal() else []) for n in selected})+'\n'
    code+='VARIABLES='+repr({n:{k:asdict(v) for k,v in allprocs[n].variables.items()} for n in selected})+'\n'
    if 'derf' in selected:code+='erf=derf\n'
    if 'derfc' in selected:code+='erfc=derfc\n'
    ast.parse(code);args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(code)
    print({n:{'variables':len(allprocs[n].variables),'statements':len(allprocs[n].statements)} for n in selected})


if __name__=='__main__':main()
