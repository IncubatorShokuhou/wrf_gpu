"""Original WRF cycling init on captured real-structured forecast columns.

Controlled incoming QKE exercises both lifecycle branches; all other column
profiles/fluxes come from the archived workload fixture. Truth is pristine
WRF mym_initialize and the original driver selection/clear block, not JAX.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

import numpy as np


def main():
    root=Path(__file__).resolve().parents[2]
    out=Path(sys.argv[1]);out.mkdir(exist_ok=False,parents=True)
    wrf=Path('<USER_HOME>/src/wrf_pristine/WRF')
    source=wrf/'phys/module_bl_mynnedmf.F'
    text=source.read_text()
    a=text.index('       IF ( (restart .or. cycling)) THEN')
    b=text.index('       dqc1',a)
    block=text[a:b]
    aliases={'qke1':'qke','qke_adv1':'adv','sh1':'work(:,1)','sm1':'work(:,2)',
             'el1':'work(:,3)','tsq1':'work(:,4)','qsq1':'work(:,5)','cov1':'work(:,6)',
             'cldfra_bl1':'blcld','qc_bl1':'blqc','qi_bl1':'blqi'}
    for old,new in aliases.items():block=re.sub(r'\b'+old+r'\b',new,block,flags=re.I)
    wrapper=(root/'tests/v025/b_phys/mynn_cold_oracle.f90').read_text()
    wrapper=wrapper.replace('kind_phys,p608,rcp,p1000mb,xlvcp,xlscp','kind_phys,p608,rcp,p1000mb,xlvcp,xlscp,zero')
    wrapper=wrapper.replace('integer :: nz,nb,col,k,kpbl','integer :: nz,nb,col,k,kpbl\nlogical :: initialize_qke,restart=.false.,cycling=.true.\nreal(kind_phys),allocatable:: adv(:),blcld(:),blqc(:),blqi(:)')
    wrapper=wrapper.replace('a(nz,13)','a(nz,14)')
    wrapper=wrapper.replace('read(10,*) ust,flt,flq,fltv,xland,ts,rhosfc',
                            'read(10,*) ust,flt,flq,fltv,xland,ts,rhosfc,dx')
    wrapper=wrapper.replace('write(20,*) nz,nb','allocate(adv(nz),blcld(nz),blqc(nz),blqi(nz))\nwrite(20,*) nz,nb')
    wrapper=wrapper.replace('qke(k)=5.*ust*max((ust*700.-zw(k))/(max(ust,.01)*700.),.01)','qke(k)=2.*a(k,6)')
    wrapper=wrapper.replace('((a(k,7)/p1000mb)**rcp)','a(k,14)')
    insertion='''work=.123;adv=.123;blcld=.123;blqc=.123;blqi=.123
'''+block+'''
 write(20,*) merge(1,0,initialize_qke),maxval(abs(work)),maxval(abs(adv)), &
              maxval(abs(blcld)),maxval(abs(blqc)),maxval(abs(blqi)),maxval(abs(qke))
 if(initialize_qke)then
  do k=1,nz
   qke(k)=5.*ust*max((ust*700.-zw(k))/(max(ust,.01)*700.),.01)
  enddo
 endif
'''
    wrapper=wrapper.replace(' call get_pblh',insertion+' call get_pblh')
    wrapper=wrapper.replace('zero_arrays(:,5),.true.,0','zero_arrays(:,5),initialize_qke,0')
    wrapper=wrapper.replace('write(20,*) qke(k)','write(20,*) qke(k),work(k,5),work(k,3)')
    (out/'oracle.f90').write_text(wrapper)
    objects=[wrf/'phys/module_bl_mynnedmf.o',wrf/'phys/module_bl_mynnedmf_common.o',
             wrf/'share/module_model_constants.o',wrf/'phys/ccpp_kind_types.o']
    cmd=['<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran','-I'+str(wrf/'phys'),'-ffree-line-length-none',
         'oracle.f90',*map(str,objects),'-o','oracle']
    built=subprocess.run(cmd,cwd=out,capture_output=True,text=True,timeout=60)
    (out/'compile.log').write_text(built.stdout+built.stderr);built.check_returncode()
    fixture=root/'tests/v025/b_phys/fixtures/mynn_closure_carries_wrf.json'
    capture=json.loads(fixture.read_text())
    columns=[]
    for row in capture['rows']:
        state,flux=row['state'],row['flux']
        for col in range(len(state['theta'])):
            for mode in ('cold','background'):
                arrays=np.stack([state[n][col] for n in ('u','v','w','theta','qv','tke','p','rho','dz','qc','qi','qs','qsq','exner')],axis=-1).astype(np.float32)
                arrays[:,5]=0 if mode=='cold' else .5
                scalars=[flux[n][col] for n in ('ustar','theta_flux','qv_flux','fltv','xland','t_skin','rhosfc')]+[row['dx']]
                columns.append((row['domain'],col,mode,arrays,np.asarray(scalars,np.float32)))
    lines=[f'44 {len(columns)} 18 3000']
    for _,_,_,array,scalars in columns:
        lines.append(' '.join(f'{x:.9e}' for x in scalars))
        lines.extend(' '.join(f'{x:.9e}' for x in layer) for layer in array)
    (out/'input.txt').write_text('\n'.join(lines)+'\n')
    result=subprocess.run([str(out/'oracle'),str(out/'input.txt'),str(out/'output.txt')],capture_output=True,text=True,timeout=60)
    (out/'run.log').write_text(result.stdout+result.stderr);result.check_returncode()
    output=(out/'output.txt').read_text().splitlines();index=1;records=[];expected=[]
    for domain,col,mode,array,scalars in columns:
        cleared=np.array(list(map(float,output[index].split())));index+=1
        pblh=float(output[index]);index+=1
        values=np.array([list(map(float,line.split())) for line in output[index:index+44]],np.float32);index+=44
        assert np.isfinite(values).all()
        assert np.all(cleared[1:]==0),'original driver failed clear control'
        assert bool(cleared[0])==(mode=='cold')
        if mode=='background':assert np.all(values[:,0]==0),'WRF cycling high-QKE quirk changed'
        records.append({'domain':domain,'column':col,'mode':mode,'initialize_qke':bool(cleared[0]),'clear_values_max':float(cleared[1:].max()),
                        'qke_max':float(values[:,0].max()),'qsq_max':float(values[:,1].max()),'el_max':float(values[:,2].max()),
                        'pblh':pblh,'source_qc_max':float(array[:,9].max()),'source_qi_max':float(array[:,10].max())})
        expected.append(values)
    np.savez(out/'expected.npz',values=np.stack(expected))
    report={'scope':'original init only, controlled incoming QKE on real-structured forecast profiles; rmol explicitly0 as retained reference',
            'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'selection_clear_source_sha256':hashlib.sha256(text[a:b].encode()).hexdigest(),
            'fixture':str(fixture),'fixture_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),
            'objects':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in objects},'command':cmd,'records':records}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'oracle').unlink()
    print({'columns':len(records),'background_zero_qke':all(r['qke_max']==0 for r in records if r['mode']=='background'),
           'max_qsq':max(r['qsq_max'] for r in records),'max_el':max(r['el_max'] for r in records)})


if __name__=='__main__':main()
