"""Reuse sealed original Fortran CONVR observations for its ICWMRDP return.

camzm_driver.F:722 assigns ICWMRDP3D directly from ql8. This fixture carries
the original observed ql8, not a previous GPU implementation. No repeat of
the unchanged convection science or original compiler build is needed.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'proofs/v040/cuplus'))
from compare_kfcup_parcel import parse


def main():
    source=Path('<USER_HOME>/wrf_gpu2_lanes/s2-cuplus/ZM08-core-original.json')
    original=json.loads(source.read_text())
    for name,expected in original['source_object_sha256'].items():
        if name.endswith('.F'):
            path=Path('<USER_HOME>/src/wrf_pristine/WRF')/name
            assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
    bank={};rows=[]
    controls=('DT','DX','CUDT','ITIMESTEP','PSFC','HT','PBLH','KPBL','HFX','QFX','UST','MAVAIL','XLAND','TSK','TPERT_CAMUWPBL')
    names=('T_PHY','QV','P','P8W','PI_PHY','U_PHY','V_PHY','Z','Z_AT_W','TKE_PBL','TH','QC','QI','CLDFRA','DZ8W')
    for index,row in enumerate(original['records']):
        fields,scalars=parse(row['stdout']);core,_=parse(row['stderr'])
        assert row['returncode']==0
        n=len(fields['IN_T_PHY']);prefix=f'row{index}'
        for name in names:
            bank[prefix+'_'+name]=np.asarray(fields.get('IN_'+name,np.zeros(n)),np.float32)
        if 'CTRL_DT' in scalars:
            ctrl=[scalars['CTRL_'+name] for name in controls]
        else:
            # Frozen original wrapper controls, also used by its already-BYTE
            # full-CPS gate (compare_camzm_cps.py). The older observer predates
            # explicit CTRL_* dumps; original input profiles remain observed.
            ctrl=[54.,9000.,0.,1.,100800. if row['case']==3 else 100000.,
                  0.,1500.,6.,150.,1.e-4,.3,1.,1.,fields['IN_T_PHY'][0],.3]
        bank[prefix+'_controls']=np.asarray(ctrl,np.float32)
        # Earlier observers dumped only n bottom interfaces. Their independent
        # full CONVR observation provides the exact missing top interface.
        for name,observed in (('P8W','paph'),('Z_AT_W','zi')):
            value=np.asarray(core['ZM_CORE_IN_'+observed],np.float32)[::-1]
            np.testing.assert_array_equal(bank[prefix+'_'+name],value[:len(bank[prefix+'_'+name])])
            bank[prefix+'_'+name]=value
        truth=np.asarray(core['ZM_CORE_OUT_ql'],np.float64)[::-1].astype(np.float32)
        assert truth.shape==(n,) and np.isfinite(truth).all()
        bank[prefix+'_ICWMRDP']=truth
        rows.append(dict(index=index,case=row['case'],pbl=row['pbl'],surface=row['surface'],
            nz=n,active=int(np.count_nonzero(truth)),
            stdout_sha256=hashlib.sha256(Path(row['stdout']).read_bytes()).hexdigest(),
            stderr_sha256=hashlib.sha256(Path(row['stderr']).read_bytes()).hexdigest()))
    output=ROOT/'proofs/s2_cammg/fixtures/camzm_handoff.npz'
    np.savez_compressed(output,**bank)
    manifest=dict(reference='Original CPU-WRF complete CAM-ZM CONVR observed ql8 -> REAL ICWMRDP3D; driver:722',
        source_receipt=str(source),source_receipt_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_sha256={k:v for k,v in original['source_object_sha256'].items() if k.endswith('.F')},
        rows=rows,asset_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
