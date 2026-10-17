"""Actual pristine radiation_driver with real WN3 columns and has_req111/000.

Generate the large Fortran call from its own declaration contract. Only the
harness is generated: the driver and RRTMG are linked from the retained WRF
library, never edited or run in the pristine source directory. solve_em is
not invoked. Oracle dumps are disabled; all outputs live in the lane build.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
WRF=ROOT/'data/wrf_pristine/WRF'
WORK=Path(sys.argv[1]).resolve();WORK.mkdir(parents=True,exist_ok=True)
ENV=Path('<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build')
FC=ENV/'bin/x86_64-conda-linux-gnu-gfortran'
source=WRF/'phys/module_radiation_driver.F'
text=subprocess.check_output([str(FC),'-E','-cpp','-DEM_CORE=1','-P','-ffree-form',str(source)],text=True)
statements=[];statement=''
for line in text.splitlines():
    line=line.split('!')[0].strip()
    if not line or line.startswith('#'):continue
    statement+=' '+line.lstrip('&').rstrip('&')
    if not line.endswith('&'):statements.append(statement.strip());statement=''
sig=next(s for s in statements if re.match(r'SUBROUTINE radiation_driver\s*\(',s,re.I))
args=[v.strip().lower() for v in sig[sig.index('(')+1:-1].split(',')]
def split(s):
    depth=0;start=0;out=[]
    for i,c in enumerate(s):
        depth+=(c=='(')-(c==')')
        if c==',' and depth==0:out.append(s[start:i].strip());start=i+1
    return out+[s[start:].strip()]
D={}
for s in statements:
    if '::' not in s:continue
    left,right=s.split('::',1)
    if not re.match(r'(real|integer|logical)\b',left,re.I):continue
    for name in split(right):
        name=re.split(r'[=(]',name)[0].strip().lower()
        if name in args and name not in D:D[name]=left
assert set(args)<=D.keys()
required=[a for a in args if 'optional' not in D[a].lower()]
extras=['p_top','yr','declinx','solconx','coszen','hrang','cldfra','qv','qc','qr','qi','qs','qg','qh',
        'curr_secs','radtacttime','adapt_step_flag','f_qi2','f_qi3','f_qv','f_qc','f_qr','f_qi','f_qs','f_qg','f_qh','olr','lwupb','lwupt','swupt','swdnb','lwdnb','sf_surface_physics']
# WRF's grid caller supplies these controls even when declared OPTIONAL;
# several original conditions read them without PRESENT/short-circuit guards.
extras += [a for a in args if a not in required+extras]
selected=required+extras
assert len(set(selected))==len(selected)
params=dict(nz=44,ids=1,ide=2,jds=1,jde=2,kds=1,kde=45,ims=1,ime=1,jms=1,jme=1,kms=1,kme=45,kts=1,kte=44,
            num_tiles=1,alevsiz=1,levsiz=59,no_src_types=1,n_ozmixm=12,n_aerosolc=1,paerlev=1,cam_abs_dim1=1,cam_abs_dim2=1)
declarations=['integer,parameter::'+','.join(f'{k}={v}' for k,v in params.items()),'integer::k,case,mpidx,use_re,ic,mpcodes(3)=[18,24,28]']
initial=[]
for name in selected:
    if name in params:continue
    attrs=[s for s in split(D[name]) if not s.lower().startswith(('intent','optional'))]
    declarations.append(','.join(attrs)+' :: '+name)
    value='.false.' if attrs[0].lower().startswith('logical') else '0'
    initial.append(f'{name}={value}')
known={'lw_physics':4,'sw_physics':4,'icloud':1,'cldovrlp':2,'idcor':0,'itimestep':1,'julday':59,'julyr':2026,'yr':2026,
       'nphs':1,'stepra':1,'ra_call_offset':0,'id':1,'cu_physics':0,'swint_opt':0,'sf_surface_physics':4,
       'aer_type':1,'wif_input_opt':1,'radt':30.,'dt':54.,'xland':2.,'xlat':28.,'xlong':-16.,'cen_lat':28.,'tsk':290.,'emiss':.98,'albedo':.08,
       'gmt':0.,'julian':59.5,'xtime':720.,'p_top':float(np.load(ROOT/'tests/v025/fid_q2/fixtures/rrtmg_mp_re_0227_v1.npz')['in_p_top']),'cam_abs_freq_s':21600.,'aer_ssa_val':1.,'aer_angexp_val':1.3,
       'aer_asy_val':.85,'i_start':1,'i_end':1,'j_start':1,'j_end':1}
initial += [f'{n}={v}' for n,v in known.items()]
initial += [f'f_{n}=.true.' for n in ('qv','qc','qr','qi','qs','qg','qh')]
read_names=['t','p','pi','rho','qv','qc','qi','qs','qr','qg','re_cloud','re_ice','re_snow']
initial += [f'read(10){n}(1,1:nz,1)' for n in read_names]
initial += ['read(10)p8w,t8w,dz8w','read(10)tsk,emiss,albedo,xland,xlat,xlong,julian,xtime']
initial += ['do k=1,kme','z(1,k,1)=sum(dz8w(1,1:k,1))','enddo']
initial += [f'{n}(1,kme,1)={n}(1,nz,1)' for n in read_names]
program='''program full_radiation
use module_radiation_driver,only:radiation_driver
use module_ra_rrtmg_lw,only:rrtmg_lwinit
use module_ra_rrtmg_sw,only:rrtmg_swinit
implicit none
'''+ '\n'.join(declarations)+'''
open(20,file='output.bin',access='stream',form='unformatted',status='replace',convert='native')
do ic=1,3
do mpidx=1,3
do use_re=0,1
open(10,file='input.bin',access='stream',form='unformatted',status='old',convert='native')
do case=1,ic
'''+ '\n'.join(initial)+'''
enddo
close(10)
mp_physics=mpcodes(mpidx)
has_reqc=use_re;has_reqi=use_re;has_reqs=use_re
if (ic==1.and.mpidx==1.and.use_re==0)then
call rrtmg_lwinit(p_top,.true.,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,i_start(1),i_end(1),j_start(1),j_end(1),kts,kte)
call rrtmg_swinit(.true.,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,i_start(1),i_end(1),j_start(1),j_end(1),kts,kte)
endif
call radiation_driver( &
'''+ ', &\n'.join(f'{n}={n}' for n in selected)+''')
write(20)ic,mp_physics,use_re
write(20)glw,swdown,gsw,olr,solconx,coszen,lwupt,swupt,lwupb
write(20)cldfra(1,1:nz,1),rthratenlw(1,1:nz,1),rthratensw(1,1:nz,1)
write(20)re_cloud(1,1:nz,1),re_ice(1,1:nz,1),re_snow(1,1:nz,1)
enddo
enddo
enddo
close(20)
end program
'''
(WORK/'driver.f90').write_text(program)
flags=['-w','-ffree-line-length-none','-fconvert=big-endian','-frecord-marker=4','-O0','-g','-fcheck=all']
incs=[f'-I{WRF/p}' for p in ('phys','frame','share','main','inc')]
(WORK/'module_radiation_driver.F').write_bytes(source.read_bytes())
subprocess.run([str(FC),'-c',*flags,*incs,'-cpp','-DEM_CORE=1','-ffree-form','module_radiation_driver.F','-o','module_radiation_driver.o'],cwd=WORK,check=True)
subprocess.run([str(FC),'-c',*flags,*incs,'driver.f90','-o','driver.o'],cwd=WORK,check=True)
libs=[WRF/p for p in ('main/libwrflib.a','external/fftpack/fftpack5/libfftpack.a','external/io_grib1/libio_grib1.a','external/io_grib_share/libio_grib_share.a','external/io_int/libwrfio_int.a')]
cmd=[str(FC),*flags,'driver.o','module_radiation_driver.o',*map(str,libs),f'-L{WRF}/external/esmf_time_f90','-lesmf_time',str(WRF/'frame/module_internal_header_util.o'),str(WRF/'frame/pack_utils.o'),f'-L{WRF}/external/io_netcdf','-lwrfio_nf',f'-L{ENV}/lib','-lnetcdff','-lnetcdf',f'-Wl,-rpath,{ENV}/lib','-o','driver']
subprocess.run(cmd,cwd=WORK,check=True)
for name in ('RRTMG_LW_DATA','RRTMG_SW_DATA','CAMtr_volume_mixing_ratio'):
    link=WORK/name
    if not link.exists():link.symlink_to(WRF/'run'/name)
fpath=ROOT/'tests/v025/fid_q2/fixtures/rrtmg_mp_re_0227_v1.npz'
fx=np.load(fpath)
columns=(31,78,114)
mapping={'t':'in_t','p':'in_p_hyd','pi':'in_pi','rho':'in_rho','qv':'in_qv','qc':'in_qc_rad','qi':'in_qi_rad','qs':'in_qs','qr':'in_qr','qg':'in_qg',
         're_cloud':'col_re_cloud','re_ice':'col_re_ice','re_snow':'col_re_snow'}
with (WORK/'input.bin').open('wb') as stream:
    for idx in columns:
        for name in read_names:np.asarray(fx[mapping[name]][idx],np.float32).tofile(stream)
        for name in ('in_p_hyd_w','in_t8w','in_dz8w'):np.asarray(fx[name][idx],np.float32).tofile(stream)
        np.asarray([fx[name][idx] for name in ('in_tsk','in_emiss','in_albedo','in_xland','in_xlat','in_xlong','in_julian')]+[fx['in_tau'][idx]*60.],np.float32).tofile(stream)
env=os.environ.copy()
for key in list(env):
    if key.startswith('WRFGPU2_ORACLE'):env.pop(key)
subprocess.run([str(WORK/'driver')],cwd=WORK,env=env,check=True)
raw=(WORK/'output.bin').read_bytes();offset=0;rows=[]
for _ in range(18):
    header=np.frombuffer(raw,offset=offset,count=3,dtype='<i4');offset+=12
    scalar=np.frombuffer(raw,offset=offset,count=9,dtype='<f4');offset+=36
    profiles=np.frombuffer(raw,offset=offset,count=6*44,dtype='<f4').reshape(6,44);offset+=6*44*4
    assert np.isfinite(scalar).all() and np.isfinite(profiles).all()
    rows.append(dict(column=int(header[0]),mp=int(header[1]),use_mp_re=int(header[2]),scalars=scalar.tolist(),profiles=profiles.tolist()))
assert offset==len(raw)
receipt=dict(scope='actual full pristine radiation_driver calls; real WN3 three columns; held radii producer qualified separately',
             source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),library_sha256=hashlib.sha256((WRF/'main/libwrflib.a').read_bytes()).hexdigest(),
             generated_driver_sha256=hashlib.sha256(program.encode()).hexdigest(),real_operand_fixture_sha256=hashlib.sha256(fpath.read_bytes()).hexdigest(),
             input_columns=list(columns),mapping=mapping,caller_controls=known,scalars=['glw','swdown','gsw','olr','solcon','coszen','lwupt','swupt','lwupb'],
             profiles=['cldfra','hlw','hsw','re_cloud','re_ice','re_snow'],rows=rows)
(ROOT/'proofs/s1_mpfix/fixtures/full_radiation_driver.json').write_text(json.dumps(receipt,separators=(',',':'))+'\n')
print('PASS actual full radiation_driver18 calls,finite scalar/profile outputs')
