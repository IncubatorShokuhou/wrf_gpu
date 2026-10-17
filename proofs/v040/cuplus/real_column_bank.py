"""Supply sealed REAL input profiles to unmodified original WRF callers.

Only the standalone driver changes. Each decimal has enough significant digits
to round trip REAL32; the candidate consumes independently dumped caller inputs.
"""
from pathlib import Path
import hashlib
import json
import numpy as np


KEYS=('T','QV','P','DZ','RHO','W0AVG','U','V','PI','TH','Z','W','ZW','P8W','QC','QI','CLDFRA','TKE')


def prepare(path,build):
    path=Path(path).resolve()
    with np.load(path,allow_pickle=False) as raw:
        cases=sorted({int(key.split('_',1)[0]) for key in raw.files})
        n=len(raw['0_T'])
        for case in cases:
            values=[raw[f'{case}_CTRL']]+[raw[f'{case}_{key}'] for key in KEYS]
            assert all(np.isfinite(v).all() and v.dtype==np.float32 for v in values)
            (build/f'profile-{case+1}.dat').write_text(''.join(
                ' '.join(f'{float(x):.9e}' for x in value)+'\n' for value in values))
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                levels=n,cases=[case+1 for case in cases],controls=json.loads(path.with_suffix('.json').read_text())['controls'])


def bind(text,bank,scheme):
    """Replace sounding inputs after construction, preserving original bodies."""
    n=bank['levels']
    import re
    text=re.sub(r'(call build_sounding\()case_id',r'\g<1>1',text,flags=re.I)
    text=text.replace('KX = 40',f'KX = {n}').replace('KX=40',f'KX={n}')
    decl='''  integer :: bank_unit
  character(len=64) :: bank_path
  real :: bank_ctrl(12), bank_profile(KX+1,18)
'''
    if scheme=='camzm':
        anchor='  character(len=32)::arg\n'
    else:
        anchor='  ! --- parse case id ---'
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor+decl if scheme=='camzm' else decl+anchor)
    block="  write(bank_path,'(A,I0,A)')'profile-',case_id,'.dat'\n"
    block+="  open(newunit=bank_unit,file=trim(bank_path),status='old',action='read')\n  read(bank_unit,*)bank_ctrl\n"
    for i,key in enumerate(KEYS,1):
        size='KX+1' if key in ('W','ZW','P8W') else 'KX'
        block+=f'  read(bank_unit,*)bank_profile(1:{size},{i})\n'
    block+='  close(bank_unit)\n  dt=bank_ctrl(2);dx=bank_ctrl(1)\n'
    aliases=({'T':'t_phy','P':'p','DZ':'dz8w','U':'u_phy','V':'v_phy','PI':'pi_phy','Z':'z',
              'ZW':'z_at_w','TKE':'tke_pbl'} if scheme=='camzm' else
             {'P':'PCPS','DZ':'DZ8W','PI':'PII','Z':'zlev'})
    wanted=('T','QV','P','DZ','RHO','U','V','PI','TH','Z','ZW','P8W','QC','QI','TKE','CLDFRA') if scheme=='camzm' else KEYS[:13]
    for key in wanted:
        if scheme!='camzm' and key=='ZW':continue
        if scheme=='camzm' and key=='RHO':continue
        target=aliases.get(key,key)
        if scheme=='camzm' and key=='CLDFRA':target='cldfra'
        size='KX+1' if key in ('W','ZW','P8W') else 'KX'
        dest=f'{target}(1,1:{size},1)' if target!='zlev' else 'zlev'
        block+=f'  {dest}=bank_profile(1:{size},{KEYS.index(key)+1})\n'
    if scheme=='camzm':
        block+='''  psfc=bank_ctrl(11);ht=bank_ctrl(10);pblh=bank_ctrl(5);kpbl=nint(bank_ctrl(6))
  hfx=bank_ctrl(3);qfx=bank_ctrl(4);ust=bank_ctrl(7);xland=bank_ctrl(9);tsk=bank_ctrl(12)
'''
        anchor='  pcnst_runtime = 5'
        assert text.count(anchor)==1
        text=text.replace(anchor,block+anchor)
        # Retain the full interface profiles, including the top face.
        for name in ('p8w','z_at_w','cmfmc'):
            text=re.sub(r"(call dump_profile\('[^']*',"+name+r")\)",r'\1,KX+1)',text,flags=re.I)
        text=text.replace('subroutine dump_profile(name,arr)','subroutine dump_profile(name,arr,count)')
        text=text.replace('    integer::kk\n    do kk=1,KX','    integer::kk,last\n    integer,optional::count\n    last=KX\n    if(present(count))last=count\n    do kk=1,last')
        # Surface controls are independently captured rather than reconstructed
        # from a profile number in the Python comparison.
        dump=''.join(f"  write(*,'(A,ES23.15)')'CTRL_{name.upper()}=',{name}"+'\n'
                     for name in ('dt','dx','cudt','itimestep'))
        dump+=''.join(f"  write(*,'(A,ES23.15)')'CTRL_{name.upper()}=',{name}(1,1)\n" for name in
                      ('psfc','ht','pblh','kpbl','hfx','qfx','ust','mavail','xland','tsk','tpert_camuwpbl'))
        text=text.replace("  write(*,'(A,I0)')'CASE=',case_id",dump+"  write(*,'(A,I0)')'CASE=',case_id")
        # Integer diagnostics must be converted for an ES format.
        text=text.replace("'CTRL_ITIMESTEP=',itimestep","'CTRL_ITIMESTEP=',real(itimestep)")
        text=text.replace("'CTRL_KPBL=',kpbl(1,1)","'CTRL_KPBL=',real(kpbl(1,1))")
    else:
        block+='  BACKGROUND_WIND=0.\n'
        anchor='  DO jj=jms,jme'
        assert text.count(anchor)==1
        text=text.replace(anchor,block+anchor)
        surface='''  HFX=bank_ctrl(3);PBLH=bank_ctrl(5);XLAND=bank_ctrl(9)
'''
        if scheme=='mskf':
            surface+='  UST=bank_ctrl(7);ZOL=bank_ctrl(8)\n'
            anchor='  U(:,kts:kte,:)=U(:,kts:kte,:)+BACKGROUND_WIND\n'
            text=text.replace(anchor,anchor+surface)
        else:
            surface+='''  QFX=bank_ctrl(4);KPBL=nint(bank_ctrl(6));HT=bank_ctrl(10);PSFC=bank_ctrl(11);TSK=bank_ctrl(12)
  DO jj=jms,jme
    DO ii=ims,ime
      ZW(ii,1:KX+1,jj)=bank_profile(1:KX+1,13)
    ENDDO
  ENDDO
'''
            anchor="  CALL dump_col('IN_W0AVG',W0AVG)"
            assert text.count(anchor)==1
            text=text.replace(anchor,surface+anchor)
        # The original builder's top W=0 assignment must not erase actual W.
        text=text.replace('      W(ii,KX+1,jj)=0.','      W(ii,KX+1,jj)=bank_profile(KX+1,12)')
        before='  ! ---------------- dump everything (flat key=value) -----------------'
        dump=''.join(f"  WRITE(*,'(A,ES23.15)')'CTRL_{name}=',bank_ctrl({index})\n" for index,name in
                     ((1,'DX'),(2,'DT'),(3,'HFX'),(4,'QFX'),(5,'PBLH'),(6,'KPBL'),(7,'UST'),
                      (8,'ZOL'),(9,'XLAND'),(10,'HT'),(11,'PSFC'),(12,'TSK')))
        assert text.count(before)==1
        text=text.replace(before,dump+before)
    return text
