"""Independent original-WRF contraction and component-wise input-ULP controls.

Restores raw REAL4 arguments from a big-endian stream, so recompiling the caller
cannot change the operands. Original microphysics bodies are not edited. Large
streams/builds are disposable; the reduced enclosure and audit are retained.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from build_reference import FC,WRF,MODULES,wrapper


def freeze_initialized(source,module,values):
    """Append an assignment-only setter; original routine bodies stay intact."""
    from port import lines,declaration
    variables={};in_type=False
    contains=re.search(r'^\s*contains\s*$',source,re.M|re.I)
    end=re.search(r'^\s*end\s+module\s+'+module+r'\b',source,re.M|re.I)
    header_end=contains.start() if contains else end.start()
    for _,s in lines(source[:header_end]):
        if re.match(r'type\s+\w+',s):in_type=True;continue
        if s.startswith('end type'):in_type=False;continue
        if not in_type:declaration(s,variables)
    setter='\nsubroutine freeze_cammg_'+module+'\n'
    getter=''
    for name,value in values.items():
        spec=variables[name]
        if not spec.parameter:
            a=np.asarray(value)
            def literal(x):
                if spec.kind=='bool':return '.true.' if x else '.false.'
                if spec.kind=='i32':return str(int(x))
                return repr(float(x))+'_r8'
            if a.ndim:setter+=f'{name}=reshape(['+','.join(literal(x) for x in a.reshape(-1,order='F'))+f'],shape({name}))\n'
            else:setter+=f'{name}={literal(a)}\n'
        expression=f'merge(1._r8,0._r8,{name})' if spec.kind=='bool' else f'real({name},r8)' if spec.kind=='i32' else name
        getter+=f"write(*,'(A,*(1X,ES26.17E3))') 'FROZEN {module}.{name}',{expression}\n"
    setter+=getter+'end subroutine\n'
    public='\npublic::freeze_cammg_'+module+'\n'
    source=source[:header_end]+public+('' if contains else '\ncontains\n')+source[header_end:]
    end=re.search(r'^\s*end\s+module\s+'+module+r'\b',source,re.M|re.I)
    return source[:end.start()]+setter+source[end.start():]


def compile_stream(build,mode):
    build.mkdir(parents=True,exist_ok=True)
    text=wrapper()
    modules={'module_cam_mp_cldwat2m_micro':'cldwat2m_micro','module_cam_mp_ndrop':'ndrop',
             'module_cam_mp_microp_aero':'microp_aero','module_data_cam_mam_aero':'modal_aero_data'}
    frozen={'cldwat2m_micro':{k:np.float64(v) for k,v in json.loads(Path('proofs/s2_cammg/CONSTANTS.json').read_text())['constants'].items()}}
    with np.load('proofs/s2_cammg/fixtures/aerosol_constants.npz') as z:
        namespaces={'ndrop':'ndrop','aero':'microp_aero','modal':'modal_aero_data'}
        for key in z.files:
            ns,name=key.split('.');frozen.setdefault(namespaces[ns],{})[name]=z[key]
    text=text.replace('  use module_mp_cammgmp_driver, only:',
        ''.join(f'  use {m},only:freeze_cammg_{m}\n' for m in modules.values())+'  use module_mp_cammgmp_driver, only:')
    declaration=text[text.index('  real :: '):text.index('  pcnst_runtime=12')]
    three=re.findall(r'(\w+)\(1,nz\+1,1\)',declaration)
    two=re.findall(r'(\w+)\(1,1\)',declaration)
    fields=three+two
    assert len(fields)==52 and len(set(fields))==52
    callstart=text.index('      call CAMMGMP(');callend=text.index("      write(*,'(A)') 'OUTPUT'",callstart)
    call=text[callstart:callend]
    start=text.index('  do ic=1,18');end=text.index('\ncontains\n',start)
    executable=''.join(f'  call freeze_cammg_{m}\n' for m in modules.values())+'''  open(10,file='inputs.bin',access='stream',form='unformatted',status='old')
  open(20,file='outputs.bin',access='stream',form='unformatted',status='replace')
  read(10) records
  do record=1,records
    read(10) ic,itimestep,config_flags%cu_physics,config_flags%shcu_physics
    read(10) dt,accum_mode,aitken_mode,coarse_mode
'''+''.join(f'    read(10) {n}\n' for n in fields)+call+''.join(f'    write(20) {n}\n' for n in fields)+'''  end do
  close(10);close(20)
'''
    text=text[:start]+executable+text[end:]
    text=text.replace('  integer :: k,m,mm,ic,itimestep,mode','  integer :: k,m,mm,ic,itimestep,mode,records,record')
    (build/'driver.f90').write_text(text)
    stubs=Path('proofs/v034/camuw_oracle/wrf_runtime_stubs.f90').read_text()+'''
subroutine wrf_error_fatal(msg)
character(len=*),intent(in)::msg
write(0,'(A)')trim(msg)
stop 2
end subroutine
'''
    (build/'stubs.f90').write_text(stubs)
    flags=['-O2','-ffree-form','-ffree-line-length-none','-fconvert=big-endian','-frecord-marker=4',
           '-finit-real=zero','-finit-integer=0','-finit-logical=false','-mfma',
           '-ffp-contract='+('off' if mode=='off' else 'fast'),f'-I{WRF}/phys',f'-I{WRF}/share',f'-I{WRF}/frame']
    rebuilt=[MODULES[i] for i in (7,8,6,9,5,4,3,2,1,0)]
    for name in rebuilt:
        source=subprocess.check_output(['cpp','-P','-traditional-cpp','-DWRF_CHEM=0','-DEM_CORE=1',str(WRF/'phys'/f'{name}.F')],text=True)
        if name in modules:source=freeze_initialized(source,modules[name],frozen[modules[name]])
        (build/f'{name}.f90').write_text(source)
        subprocess.run([str(FC),*flags,'-c',str(build/f'{name}.f90'),'-o',str(build/f'{name}.o')],cwd=build,check=True)
    for name in ('stubs','driver'):
        subprocess.run([str(FC),*flags,'-c',str(build/f'{name}.f90'),'-o',str(build/f'{name}.o')],cwd=build,check=True)
    objects=[build/f'{n}.o' if n in rebuilt else WRF/'phys'/f'{n}.o' for n in MODULES]
    objects+=[WRF/'share/module_model_constants.o',WRF/'frame/module_state_description.o']
    subprocess.run([str(FC),str(build/'driver.o'),str(build/'stubs.o'),*map(str,objects),'-lmvec','-lm','-o',str(build/'oracle.exe')],check=True)
    asm=subprocess.check_output(['objdump','-d',*[str(build/f'{n}.o') for n in rebuilt]],text=True)
    fma=sum(bool(re.search(r'\bv(?:f(?:m|n)(?:add|sub)|fmaddsub|fmsubadd)',line)) for line in asm.splitlines())
    return fields,dict(flags=flags,fma_opcodes=fma,objects={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in objects}),frozen


def configurations(case):
    cu,shcu=(7,2) if case>=19 else (0,0)
    if case==20:shcu=0
    if case==21:cu=0
    if case in (22,24):cu,shcu=1,1
    modes=(1.e7,1.e6,1.e4) if case==16 else (3.e9,1.e9,5.e6) if case==17 else (0.,0.,0.) if case==18 else (1.e9,3.e8,2.e5)
    return cu,shcu,modes


def write_inputs(path,fields,bank,plan):
    selected=plan['array_arguments']+plan['scalar_arguments']
    count=0;descriptors=[];sizes=[bank['c01_s1_input_'+n].size for n in fields]
    offsets=np.cumsum([0,*sizes]);records=[]
    for case in range(1,25):
        for step in (1,2):
            prefix=f'c{case:02d}_s{step}';values=np.concatenate([bank[prefix+'_input_'+n] for n in fields])
            controls=[(-1,-1,0)]
            for name in selected:
                if name=='dt':controls.extend([(-2,0,-1),(-2,0,1)]);continue
                index=fields.index(name)
                controls.extend((index,cell,direction) for cell in range(sizes[index]) for direction in (-1,1))
            records.append((case,step,prefix,values,controls));count+=len(controls)
    with path.open('wb') as output:
        output.write(np.asarray([count],dtype='>i4').tobytes())
        for case,step,prefix,values,controls in records:
            cu,shcu,modes=configurations(case);begin=len(descriptors)
            for arg,cell,direction in controls:
                output.write(np.asarray([case,step,cu,shcu],dtype='>i4').tobytes())
                scalars=np.asarray([18.,*modes],dtype=np.float32)
                current=values
                if arg==-2:scalars[0]=np.nextafter(scalars[0],np.float32(np.inf*direction))
                elif arg>=0:
                    current=values.copy();i=offsets[arg]+cell
                    current[i]=np.nextafter(current[i],np.float32(np.inf*direction))
                output.write(scalars.astype('>f4').tobytes());output.write(current.astype('>f4').tobytes())
                descriptors.append((case,step,arg,cell,direction))
    spans=[];at=0
    for case,step,prefix,values,controls in records:spans.append((prefix,at,at+len(controls)));at+=len(controls)
    return offsets,spans,np.asarray(descriptors,dtype=np.int32)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('output',type=Path)
    args=ap.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    plan=json.loads(Path('proofs/s2_cammg/CONTROL_PLAN.json').read_text())
    with np.load('proofs/s2_cammg/fixtures/convection.npz') as z:bank={k:z[k] for k in z.files}
    fields,off,frozen=compile_stream(args.root/'off','off')
    offsets,spans,descriptors=write_inputs(args.root/'inputs.bin',fields,bank,plan)
    records=len(descriptors);nvalues=int(offsets[-1]);input_sha=hashlib.sha256((args.root/'inputs.bin').read_bytes()).hexdigest()
    output={};receipts={};lower={};upper={}
    for mode in ('off','on'):
        build=args.root/mode
        if mode=='on':fields2,receipt,frozen2=compile_stream(build,mode);assert fields2==fields
        else:receipt=off
        (build/'inputs.bin').symlink_to(args.root/'inputs.bin')
        with (build/'stdout.txt').open('wb') as log:subprocess.run([str(build/'oracle.exe')],cwd=build,stdout=log,check=True,timeout=300)
        actual={}
        for line in (build/'stdout.txt').read_text().splitlines():
            if line.startswith('FROZEN '):
                parts=line.split();actual[parts[1]]=np.asarray(list(map(float,parts[2:])))
        for namespace,values0 in frozen.items():
            for name,value in values0.items():
                a=np.asarray(value);read=actual[namespace+'.'+name].astype(a.dtype).reshape(a.shape,order='F')
                assert read.tobytes()==a.tobytes(),('frozen-table',mode,namespace,name)
        receipt['all_initialized_values_BYTE']=True
        values=np.memmap(build/'outputs.bin',dtype='>f4',mode='r',shape=(records,nvalues))
        assert np.all(np.isfinite(values)),mode
        for prefix,begin,end in spans:
            for i,name in enumerate(fields):
                block=values[begin:end,offsets[i]:offsets[i+1]].astype(np.float32)
                key=prefix+'_'+name
                if mode=='off':
                    assert block[0].tobytes()==bank[prefix+'_output_'+name].tobytes(),('baseline',prefix,name)
                    lower[key]=block.min(axis=0);upper[key]=block.max(axis=0)
                else:lower[key]=np.minimum(lower[key],block.min(axis=0));upper[key]=np.maximum(upper[key],block.max(axis=0))
        receipts[mode]=receipt
        del values
        # Retain only small receipts; exact original sources already sealed.
        for path in build.iterdir():
            if path.suffix in ('.o','.mod','.exe','.bin'):path.unlink()
    assert receipts['off']['fma_opcodes']==0 and receipts['on']['fma_opcodes']>0,receipts
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**{'lo_'+k:v for k,v in lower.items()},**{'hi_'+k:v for k,v in upper.items()})
    summary=dict(records_per_arm=records,case_steps=48,fields=fields,finite=True,unperturbed_noFMA_BYTE=True,
                 raw_input_sha256=input_sha,plan_sha256=hashlib.sha256(Path('proofs/s2_cammg/CONTROL_PLAN.json').read_bytes()).hexdigest(),
                 descriptor_sha256=hashlib.sha256(descriptors.tobytes()).hexdigest(),arithmetic=receipts,
                 asset_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    args.output.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.root/'inputs.bin').unlink()
    print(json.dumps({k:v for k,v in summary.items() if k not in ('arithmetic','fields')}))


if __name__=='__main__':main()
