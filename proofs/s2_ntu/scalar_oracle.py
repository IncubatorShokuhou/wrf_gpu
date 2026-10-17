"""Independent pristine scalar calls, using frozen full-caller NTU operands."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import numpy as np

from build_oracle import FC,NK,CASES,VOLUME,SURFACE
from port_scalar import statements,split,ROUTINES,SOURCE


def make_driver(signatures,trace=None):
    names=sorted(set(sum(signatures.values(),[])))
    text=["program scalar_oracle","use module_mp_ntu","implicit none",
          "integer::u,o,ri,case_id,k", "real::enabled,"+",".join(names),
          "call aero_const(2)",
          "open(newunit=u,file='inputs.bin',form='unformatted',access='stream',status='old')",
          "open(newunit=o,file='outputs.bin',form='unformatted',access='stream',status='replace')"]
    if trace:text.append("open(unit=99,file='trace.txt',status='replace')")
    for name,args in signatures.items():
        text += [f"do ri=1,{len(CASES)*NK}", "read(u) enabled,"+",".join(args),
                 *( ["write(99,'(i8,1x,a,1x,i8)') 0,'row',ri"] if name==trace else [] ),
                 f"if(enabled>0.)call {name}("+",".join(args)+")", "write(o) "+",".join(args), "enddo"]
    return "\n".join(text+["close(u);close(o)","end program scalar_oracle",""])


def main():
    parser=argparse.ArgumentParser();parser.add_argument("input_npz",type=Path);parser.add_argument("out",type=Path)
    parser.add_argument("--kind",choices=("r4","r8"),default="r4")
    parser.add_argument("--trace",choices=ROUTINES)
    args=parser.parse_args()
    assert os.environ.get("JAX_PLATFORMS")=="cpu" and not Path('/tmp/wrf_gpu2_quiet').exists()
    dtype=np.dtype("float32" if args.kind=="r4" else "float64");r=dtype.type
    signatures={}
    for _,s in statements(SOURCE.read_text()):
        match=re.fullmatch(r"subroutine (\w+)\((.*)\)",s)
        if match and match[1] in ROUTINES:signatures[match[1]]=split(match[2])
    out=args.out.resolve();out.mkdir(exist_ok=False,parents=True)
    data=np.load(args.input_npz);inputs={};order=[];active={}
    for name,params in signatures.items():
        rows=[];enabled=[]
        for case_id in range(1,len(CASES)+1):
            for k in range(NK):
                v={x:r(data[f'c{case_id:02}_s0_{x}'][k]) for x in VOLUME}
                p=r(data[f'c{case_id:02}_input_p'][k]);pii=r(data[f'c{case_id:02}_input_pii'][k]);t=v['th']*pii
                v.update(tk1d=t,p1d=p,rho=p/(t*(r(1)+r(.61)*v['qv']))/r(287.058),dt=r(18),dts=r(18),sdts=r(0),
                         dtkdt=-r(.5)*r(9.806)/r(1005.46),dqvdt=r(0),dpdt=-r(9.806)*r(.5)*p/(r(287.058)*t),
                         saspr=r(1),adagr=r(1))
                v['qnin']=r(data[f'c{case_id:02}_s0_qnin'][k])
                for x in 'qv qc qr qi qs qg qh nc nr ni ns ng nh vi vs vg fi fs ai as ag ah i3m'.split():v[x+'1d']=v[x]
                v.update(i2m1d=v['ai'],s2m1d=v['as'],g2m1d=v['ag'],h2m1d=v['ah'])
                rows.append([v.get(x,r(0)) for x in params])
                mass={'solve_afac':'qc','solve_afar':'qr','solve_afai':'qi','solve_afas':'qs','solve_afag':'qg','solve_afah':'qh'}.get(name)
                enabled.append(v[mass]>r(1.e-14) if mass else (t<r(271.15) if name=='icenu' else sum(v[x] for x in 'qc qr qi qs qg qh'.split())>0))
        inputs[name]=np.asarray(rows,dtype=dtype);active[name]=np.array(enabled)
        order.append(name)
    with (out/'inputs.bin').open('wb') as f:
        for name in order:np.column_stack((active[name].astype(dtype),inputs[name])).tofile(f)
    subprocess.run([FC,'-E','-P','-cpp','-ffree-form',str(SOURCE),'-o',str(out/'original.F90')],check=True)
    if args.trace:
        # Add observations after complete original statements. No arithmetic or
        # control expression is changed; final outputs must reproduce base bytes.
        lines=[];inside=False
        raw=SOURCE.read_text().splitlines()
        skip=False
        for i,line in enumerate(raw):
            if line.strip()=='#ifdef DOUBLE_PRECISION':skip=True;raw[i]='';continue
            if line.strip()=='#else':skip=False;raw[i]='';continue
            if line.strip()=='#endif':raw[i]='';continue
            if skip:raw[i]=''
        for n,s in statements('\n'.join(raw)):
            if s.startswith('subroutine '+args.trace+'('):inside=True
            if s.startswith('end subroutine'):inside=False
            lines.append(s)
            if not inside or re.match(r'^(real|integer|logical|implicit|data)\b',s):continue
            # Observe the assigned value; single-line IF may leave its previous
            # initialized value, exactly as the functional port's masked write.
            action=s
            if re.match(r'if\s*\(',s) and not s.endswith('then'):
                depth=0
                for pos,ch in enumerate(s[s.index('('):],s.index('(')):
                    depth+=(ch=='(')-(ch==')')
                    if depth==0:action=s[pos+1:].strip();break
            writes=[]
            if re.match(r'^\w+\s*=',action):writes=[action.split('=')[0].strip()]
            elif action.startswith('call '):
                call=re.fullmatch(r'call (\w+)\((.*)\)',action)
                if call and call[1] in signatures:writes=split(call[2])
            for key in writes:lines.append(f"write(99,'(i8,1x,a,1x,es26.17)') {n},'{key}',real({key})")
            if n==1451:
                for key,value in [('base','2.*amh*g/aaw'),('basepow','(2.*amh*g/aaw)**vtb1'),('kinvpow','kinv**(1.-2.*vtb1)'),('vta1','vta1'),('vtb1','vtb1')]:
                    lines.append(f"write(99,'(i8,1x,a,1x,es26.17)') {n},'{key}',{value}")
        # Preprocess both versions the same way, retaining the original REAL4 branch.
        (out/'instrumented.F').write_text('\n'.join(lines)+'\n')
        (out/'original.F90').write_text('\n'.join(lines)+'\n')
    (out/'error.F90').write_text("module module_wrf_error\ncontains\nsubroutine wrf_error_fatal(s)\ncharacter(*)::s\nprint *,s\nerror stop 1\nend subroutine\nend module\n")
    (out/'driver.F90').write_text(make_driver(signatures,args.trace))
    cmd=[FC,'-ffree-form','-ffree-line-length-none','-O0','-ffp-contract=off','-fno-range-check','-finit-real=zero','-finit-integer=0','-finit-logical=false','error.F90','original.F90','driver.F90','-o','oracle']
    if args.kind=='r8':cmd[1:1]=['-fdefault-real-8','-fdefault-double-8']
    subprocess.run(cmd,cwd=out,check=True)
    with (out/'stdout.log').open('w') as log:subprocess.run([str(out/'oracle')],cwd=out,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    raw=np.fromfile(out/'outputs.bin',dtype=dtype);offset=0;fields={};nf={}
    for name in order:
        count=inputs[name].size;output=raw[offset:offset+count].reshape(inputs[name].shape);offset+=count
        fields[name+'_input']=inputs[name];fields[name+'_output']=output;fields[name+'_active']=active[name]
        nf[name]=int(np.count_nonzero(~np.isfinite(output[active[name]])))
    assert offset==raw.size
    np.savez_compressed(out/'scalars.npz',**fields)
    manifest={'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'input_npz_sha256':hashlib.sha256(args.input_npz.read_bytes()).hexdigest(),
              'signatures':signatures,'compile':cmd,'kind':args.kind,'samples':{n:int(active[n].sum()) for n in order},'nonfinite_active':nf,
              'npz_sha256':hashlib.sha256((out/'scalars.npz').read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
    for name in ('inputs.bin','outputs.bin','original.F90','instrumented.F','error.F90','driver.F90','oracle','module_mp_ntu.mod','module_wrf_error.mod'):(out/name).unlink(missing_ok=True)


if __name__=='__main__':main()
