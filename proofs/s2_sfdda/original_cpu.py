"""Run pristine Swiss 36s controls against actual auxinput9 files, privately.

Exact lane-owned original CPU-WRF executable; no history/reference symlinks
in writable case dirs. Thompson writable tables are copied. Refuse E104
instrumented binaries. Run under the lane's CPU affinity and timeout.
"""
from pathlib import Path
import hashlib,importlib.util,json,os,shutil,subprocess,sys,signal
from smoke import SOURCE,prepare


def main(root,modes=(0,1,2)):
    root.mkdir(parents=True,exist_ok=True)
    prepare(root/'analyses')
    exe=(SOURCE/'wrf.exe').read_bytes()
    if any(x in exe for x in (b'em_quarter_ss_center_dump',b'em_grav2d_x_touchdown_dump')):
        raise RuntimeError('E104 instrumented original WRF executable refused')
    helper=Path(__file__).parents[1]/'s1_specnudge/prepare_cpu_cases.py'
    spec=importlib.util.spec_from_file_location('surface_original_helpers',helper)
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    launcher='<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/mpirun'
    temporary=root.parent/'mpi';temporary.mkdir(exist_ok=True)
    environment=dict(os.environ,TMPDIR=str(temporary),OMPI_MCA_orte_tmpdir_base=str(temporary),
                     PRTE_MCA_prte_tmpdir_base=str(temporary))
    receipt=dict(original_executable=str((SOURCE/'wrf.exe').resolve()),executable_sha256=hashlib.sha256(exe).hexdigest(),
                 steps=2,dt_s=18.,analyses_from_original_cpu=True,wrf_bodies_modified=False,arms={})
    for mode in modes:
        case=root/f'mode{mode}';case.mkdir(exist_ok=True)
        for p in SOURCE.iterdir():
            if p.name.startswith(('wrfout','wrfrst','rsl.','namelist','wrf.exe','wrffdda','wrfsfdda','sgfdda','fort.')):
                continue
            if not p.is_file():continue
            if p.name in h.TABLE_NAMES:shutil.copyfile(p,case/p.name)
            else:(case/p.name).symlink_to(p)
        for name in h.TABLE_NAMES:
            cached=(root/'mode0'/name) if mode and receipt['arms'].get('0',{}).get('success') else h.TABLES/name
            if not (case/name).exists() and cached.is_file():shutil.copyfile(cached,case/name)
        (case/'wrf.exe').write_bytes(exe);(case/'wrf.exe').chmod(0o755)
        for name in ('wrffdda_d01','wrfsfdda_d01'):(case/name).symlink_to(root/'analyses'/name)
        text=h.edit((SOURCE/'namelist.input').read_text(),'time_control',dict(run_days=0,run_hours=0,run_minutes=0,run_seconds=36,
            end_day=15,end_hour=0,end_minute=0,end_second=36,history_interval=0,history_interval_s=18,history_interval_m=0,frames_per_outfile=1))
        text=h.edit(text,'fdda',dict(grid_fdda=1,gfdda_inname="'wrffdda_d<domain>'",gfdda_interval_m=60,gfdda_end_h=1,io_form_gfdda=2,
            guv='3e-4',gt='3e-4',gq='3e-4',grid_sfdda=mode,sgfdda_inname="'wrfsfdda_d<domain>'",sgfdda_interval_m=60,sgfdda_end_h=1,
            io_form_sgfdda=2,guv_sfc='3e-4',gt_sfc='3e-4',gq_sfc='3e-4',rinblw=100.,fgdt=.9,if_ramping=0,dtramp_min=0.))
        fields=['RUNDGDTEN','RVNDGDTEN','RTHNDGDTEN','RQVNDGDTEN']
        if mode:fields+=['U10_NDG_OLD','U10_NDG_NEW','V10_NDG_OLD','V10_NDG_NEW','TH2_NDG_OLD','TH2_NDG_NEW','Q2_NDG_OLD','Q2_NDG_NEW','ODIS_NDG_OLD','ODIS_NDG_NEW']
        (case/'surface_fields.txt').write_text('+:h:0:'+','.join(fields)+'\n')
        text=h.edit(text,'time_control',dict(iofields_filename="'surface_fields.txt'",ignore_iofields_warning='.false.'))
        (case/'namelist.input').write_text(text)
        with (case/'launcher.log').open('w') as log:
            process=subprocess.Popen([launcher,'-np','1','--bind-to','none','--mca','btl','self,tcp','--mca','pml','ob1','./wrf.exe'],
                cwd=case,stdout=log,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
            try:code=process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
                raise
        rsl=(case/'rsl.error.0000').read_text() if (case/'rsl.error.0000').exists() else ''
        receipt['arms'][str(mode)]=dict(returncode=code,success='SUCCESS COMPLETE WRF' in rsl,
            outputs=[p.name for p in sorted(case.glob('wrfout_d01*'))],namelist_sha256=hashlib.sha256(text.encode()).hexdigest())
        (root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        assert code==0 and receipt['arms'][str(mode)]['success'],(mode,rsl[-2500:])


if __name__=='__main__':main(Path(sys.argv[1]).resolve(),tuple(map(int,sys.argv[2:])) or (0,1,2))
