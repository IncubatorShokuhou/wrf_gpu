"""Compile exact pristine radius routines and caller blocks; no model-run/library writes.

Thompson calc_effectRad is compiled unchanged with its literal declarations and
only the initialization statements it reads. RRTMG's caller mediation blocks
are extracted unchanged, with EM_CORE=1, and supplied with the flags compiled
from physics_init. radiation_driver's unmodified forwarding call is sealed.
These are operator/caller gates, not a coupled radiation-forecast claim.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
WRF=ROOT/'data/wrf_pristine/WRF'
WORK=Path(sys.argv[1]).resolve(); WORK.mkdir(parents=True,exist_ok=True)
SAVE=ROOT/'proofs/s1_mpfix/fixtures'
FC='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
receipt={'sources':{},'extracts':{},'scope':'literal pristine operators + caller flags/forwarding, not whole forecast'}
def source(name):
    path=WRF/name
    receipt['sources'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
    txt=path.read_text()
    (WORK/path.name).write_text(txt)
    return txt

def build(name,program,flags=()):
    (WORK/f'{name}.F90').write_text(program)
    receipt['extracts'][name]=hashlib.sha256(program.encode()).hexdigest()
    subprocess.run([FC,'-O2','-cpp','-DEM_CORE=1','-ffree-line-length-none',*flags,f'{name}.F90','-o',name],cwd=WORK,check=True)
    return WORK/name

init=source('phys/module_physics_init.F')
registry=source('frame/module_state_description.F')
rad=source('phys/module_radiation_driver.F')
start=init.index('   has_reqc = 0');stop=init.index('! for P3,',start)
block=init[start:stop]+'   ENDIF\n'
constants=dict(re.findall(r'INTEGER, PARAMETER :: (\w+) = (\d+)',registry,re.I))
needed={w.lower() for w in re.findall(r'\b\w+\b',block)}
params='\n'.join(f'integer,parameter::{k}={v}' for k,v in constants.items() if k.lower() in needed)
program=f'''program flags
implicit none
{params}
type config_type
integer::use_mp_re=1,ra_lw_physics=4,ra_sw_physics=4,mp_physics,nssl_2moment_on=1
end type
type(config_type)::config_flags
integer::has_reqc,has_reqi,has_reqs,mp,use_re
integer,parameter::cases(6)=[8,18,24,26,28,40]
do mp=1,6
do use_re=0,1
config_flags%mp_physics=cases(mp)
config_flags%use_mp_re=use_re
{block}
write(*,*)cases(mp),use_re,has_reqc,has_reqi,has_reqs
enddo
enddo
end program
'''
exe=build('flags',program)
rows=[list(map(int,l.split())) for l in subprocess.check_output([exe],cwd=WORK,text=True).splitlines()]
receipt['flags']=rows
assert all(row[2:]==([1,1,1] if row[1] and row[0]!=40 else [0,0,0]) for row in rows)
# Seal the exact radiation-driver -> RRTMG forwarding statements as caller evidence.
forward=[]
for match in re.finditer(r'RE_CLOUD=re_cloud,RE_ICE=re_ice,RE_SNOW=re_snow,.*\n[^\n]*has_reqc=has_reqc,has_reqi=has_reqi,has_reqs=has_reqs,',rad,re.I):
    forward.append(match.group())
assert len(forward)>=4
receipt['radiation_driver_forwarding']=forward

th=source('phys/module_mp_thompson.F'); model=source('share/module_model_constants.F')
param_names=['PI','rho_w','rho_i','Nt_c','Nt_c_max','mu_i','bv_i','am_r','bm_r','am_s','bm_s','am_i','bm_i','R1','R2','R']
decls=[]
for name in param_names:
    line=re.search(r'^ *REAL, PARAMETER, PRIVATE:: '+name+r' *=.*$',th,re.M).group()
    decls.append(line.replace(', PRIVATE',''))
for name in ('sa','sb'):
    declaration=re.search(r'      REAL, DIMENSION\(10\), PARAMETER, PRIVATE:: &\n      '+name+r' = \(/.*?/\)',th,re.S).group()
    decls.append(declaration.replace(', PRIVATE',''))
for name in ('RE_QC_BG','RE_QI_BG','RE_QS_BG'):
    decls.append(re.search(r'^.*PARAMETER :: '+name+r' *=.*$',model,re.M).group())
writeback='\n'.join(re.findall(r'^ *re_(?:cloud|ice|snow)\(i,k,j\)\s*= MAX\(RE_.*$',th,re.M)[:3])
assert len(writeback.splitlines())==3
calc=re.search(r'      subroutine calc_effectRad .*?      end subroutine calc_effectRad',th,re.S|re.I).group()
wgam=re.search(r'      REAL FUNCTION WGAMMA\(y\).*?      END FUNCTION WGAMMA',th,re.S|re.I).group()
gamln=re.search(r'      REAL FUNCTION GAMMLN\(XX\).*?      END FUNCTION GAMMLN',th,re.S|re.I).group()
initial=th[th.index('      cie(1) ='):th.index('      cre(1) =')]
initial+='\n'+re.search(r'^ *obmr =.*$',th,re.M).group()+'\n'+re.search(r'^ *cse\(1\) =.*$',th,re.M).group()+'\n'+re.search(r'^ *oams =.*$',th,re.M).group()
program='module reference\nimplicit none\n'+'\n'.join(decls)+'''
logical::is_aerosol_aware=.true.
real::cie(7),cig(7),oig1,oig2,obmi,obmr,cse(17),oams
contains
'''+calc+'\n'+wgam+'\n'+gamln+'\nend module\nprogram run\nuse reference\nimplicit none\ninteger,parameter::n=12\nreal::t(n),p(n),qv(n),qc(n),nc(n),qi(n),ni(n),qs(n),re_qc1d(n),re_qi1d(n),re_qs1d(n)\nreal::re_cloud(1,n,1),re_ice(1,n,1),re_snow(1,n,1)\ninteger::k,i=1,j=1\n'+initial+'''
open(10,file='thompson_input.bin',access='stream',form='unformatted',status='old',convert='native')
read(10)t,p,qv,qc,nc,qi,ni,qs
close(10)
call calc_effectRad(t,p,qv,qc,nc,qi,ni,qs,re_qc1d,re_qi1d,re_qs1d,1,n)
do k=1,n
'''+writeback+'''
write(*,'(11ES25.16)')t(k),p(k),qv(k),qc(k),nc(k),qi(k),ni(k),qs(k),re_cloud(i,k,j),re_ice(i,k,j),re_snow(i,k,j)
enddo
end program
'''
inputs=np.array([[290,280,260,250,240,230,220,210,275,255,235,200],
                [90000,85000,80000,70000,60000,50000,40000,30000,90000,65000,45000,25000],
                [0.005]*12,[0,1e-5,1e-4,1e-3,1e-4,1e-4,1e-4,1e-5,1e-12,0,1e-3,1e-6],
                [0,1e7,1e8,1e9,1e5,1e10,1e-2,3e8,1e8,0,2e8,1e6],
                [0,0,1e-5,1e-4,1e-3,1e-6,1e-4,1e-4,1e-12,0,1e-3,1e-6],
                [0,0,1e5,1e6,1e7,1e3,1e8,1e-6,1e5,0,2e5,1e6],
                [0,0,1e-5,1e-4,1e-3,1e-6,1e-4,1e-4,1e-12,0,1e-3,1e-6]],dtype=np.float32)
inputs.tofile(WORK/'thompson_input.bin')
exe=build('thompson',program)
rows=np.array([list(map(float,l.split())) for l in subprocess.check_output([exe],cwd=WORK,text=True).splitlines()])
(SAVE/'thompson_aero_radii.json').write_text(json.dumps({'inputs':rows[:,:8].T.tolist(),'radii':rows[:,8:].T.tolist(),'names':['T','p','qv','qc','Nc','qi','Ni','qs']},indent=2)+'\n')

# Distinct microns include ocean/land liquid fallback, fractional-T ice
# interpolation, active scheme radii, and snow above the130um optics cutoff.
T=[290.5,250.25,200.75,180.2,260.9,280.,240.,220.]
cf=[1.,1.,0.,1.,1.,0.,1.,1.]
rc=[2.49e-6,2.49e-6,2.49e-6,3e-6,10e-6,20e-6,40e-6,50e-6]
ri=[4.99e-6,4.99e-6,4.99e-6,4.99e-6,10.01e-6,25e-6,100e-6,125e-6]
rs=[9.99e-6,25e-6,100e-6,130e-6,131e-6,200e-6,500e-6,999e-6]
med={'T':T,'cf':cf,'re_cloud':rc,'re_ice':ri,'re_snow':rs,'outputs':{}}
for kind in ('lw','sw'):
    text=source(f'phys/module_ra_rrtmg_{kind}.F')
    start=text.index(f'         inflg{kind} = 2',text.index('SUBROUTINE RRTMG_') if 'SUBROUTINE RRTMG_' in text else 0)
    stop=text.index('! Layer indexing',start) if kind=='lw' else text.index('! Set cosine of solar zenith angle',start)
    block=text[start:stop]
    retab_source = text if kind == 'lw' else (WRF/'phys/module_ra_rrtmg_lw.F').read_text()
    retab=re.search(r'(?i)data\s+retab\s*/.*?/',retab_source,re.S).group()
    if kind == 'sw':
        assert 'use module_ra_rrtmg_lw, only : inirad, o3data, relcalc, reicalc, retab' in text
    outputs=[]
    for land in (1.,2.):
        program=f'''program mediation
implicit none
integer,parameter::n=8,kts=1,kte=n
integer::i=1,j=1,k,ncol=1,icloud=1,has_reqc=1,has_reqi=1,has_reqs=1
integer::inflg{kind},iceflg{kind},liqflg{kind},idx_rei
real::corr,tem1,tem2,tem3,thresh,re_50C,re_40C,re_30C,re_20C
real::retab(95),recloud1D(1,n),reice1D(1,n),resnow1D(1,n)
real::re_cloud(1,n,1),re_ice(1,n,1),re_snow(1,n,1),cldfra3d(1,n,1),t3d(1,n,1),xland(1,1)
real::qi3d(1,n,1),qi1d(n),qs1d(n),rho1d(n),t1d(n)
{retab}
open(10,file='mediation_input.bin',access='stream',form='unformatted',status='old',convert='native')
read(10)t3d,cldfra3d,re_cloud,re_ice,re_snow
close(10)
xland={land}
{block}
do k=1,n
write(*,'(3ES25.16,3I4)')recloud1D(1,k),reice1D(1,k),resnow1D(1,k),inflg{kind},iceflg{kind},liqflg{kind}
enddo
end program
'''
        np.array([T,cf,rc,ri,rs],np.float32).tofile(WORK/'mediation_input.bin')
        exe=build(f'mediation_{kind}_{int(land)}',program)
        outputs.append([list(map(float,l.split())) for l in subprocess.check_output([exe],cwd=WORK,text=True).splitlines()])
    med['outputs'][kind]=outputs
(SAVE/'radiation_mediation.json').write_text(json.dumps(med,indent=2)+'\n')
(SAVE/'radius_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
