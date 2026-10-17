"""Relink a PRIVATE clean DM WRF with write-only fddaobs boundary probes.

No arithmetic statement in the original module is changed. Installed modules,
archives and main objects are read-only. Compiler outputs and executable belong
to the supplied lane directory. This diagnostic executable has no timing claim.
"""
import hashlib,json,re,shlex,subprocess,sys
from pathlib import Path

ROOT=Path('<DATA_ROOT>/canairy_meteo/artifacts/wrf_src/WRF')
BUILD=ROOT/'_build_gen2_dmpar'
CANONICAL=Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_fddaobs_driver.F')


def main(out):
    out.mkdir(parents=True,exist_ok=True)
    original=(ROOT/'phys/module_fddaobs_driver.F').read_text()
    if original!=CANONICAL.read_text():raise RuntimeError('clean DM driver differs from canonical pristine source')
    text=original.replace('! Local variables','integer :: probe_unit\ncharacter(len=80) :: probe_name\n! Local variables',1)
    entry='''
  write(probe_name,'(a,i6.6,a)') 'obs_probe_',itimestep-1,'.bin'
  open(newunit=probe_unit,file=trim(probe_name),status='replace',access='stream',form='unformatted')
  write(probe_unit) ide-ids,jde-jds,kde-kds,nstat,itimestep-1,nobs_ndg_vars,nobs_err_flds
  write(probe_unit) xtime,dt,dx
  write(probe_unit) fdob%rio(1:nstat),fdob%rjo(1:nstat),fdob%rko(1:nstat),fdob%timeob(1:nstat)
  write(probe_unit) fdob%lev_in_ob(1:nstat),fdob%nlevs_ob(1:nstat),fdob%plfo(1:nstat)
  write(probe_unit) varobs(:,1:nstat),errf(:,1:nstat)
  write(probe_unit) ub(ids:ide,kds:kde-1,jds:jde-1),vb(ids:ide-1,kds:kde-1,jds:jde)
  write(probe_unit) tb(ids:ide-1,kds:kde-1,jds:jde-1),qvb(ids:ide-1,kds:kde-1,jds:jde-1)
  write(probe_unit) pbase(ids:ide-1,kds:kde-1,jds:jde-1),pp(ids:ide-1,kds:kde-1,jds:jde-1)
  write(probe_unit) z(ids:ide-1,kds:kde-1,jds:jde-1)
  write(probe_unit) uratx(ids:ide-1,jds:jde-1),vratx(ids:ide-1,jds:jde-1),tratx(ids:ide-1,jds:jde-1)
  write(probe_unit) kpbl(ids:ide-1,jds:jde-1),regime(ids:ide-1,jds:jde-1),pblh(ids:ide-1,jds:jde-1),ht(ids:ide-1,jds:jde-1)
  write(probe_unit) mut(ids:ide-1,jds:jde-1),muu(ids:ide-1,jds:jde-1),muv(ids:ide-1,jds:jde-1)
  write(probe_unit) c1h(kds:kde-1),c2h(kds:kde-1)
  write(probe_unit) msfux(ids:ide-1,jds:jde-1),msfuy(ids:ide-1,jds:jde-1)
  write(probe_unit) msfvx(ids:ide-1,jds:jde-1),msfvy(ids:ide-1,jds:jde-1)
  write(probe_unit) msftx(ids:ide-1,jds:jde-1),msfty(ids:ide-1,jds:jde-1)
  write(probe_unit) savwt(1:4,ids:ide-1,kds:kde-1,jds:jde-1)
  write(probe_unit) ru_tendf(ids:ide-1,kds:kde-1,jds:jde-1),rv_tendf(ids:ide-1,kds:kde-1,jds:jde-1)
  write(probe_unit) t_tendf(ids:ide-1,kds:kde-1,jds:jde-1),moist_tend(ids:ide-1,kds:kde-1,jds:jde-1)
'''
    finish='''
  write(probe_unit) varobs(:,1:nstat),errf(:,1:nstat),fdob%rko(1:nstat)
  write(probe_unit) savwt(1:4,ids:ide-1,kds:kde-1,jds:jde-1)
  write(probe_unit) ru_tendf(ids:ide-1,kds:kde-1,jds:jde-1),rv_tendf(ids:ide-1,kds:kde-1,jds:jde-1)
  write(probe_unit) t_tendf(ids:ide-1,kds:kde-1,jds:jde-1),moist_tend(ids:ide-1,kds:kde-1,jds:jde-1)
  close(probe_unit)
'''
    text=text.replace('  if (.not. nudge_flag) return','  if (.not. nudge_flag) return\n'+entry,1)
    marker='  END SUBROUTINE fddaobs_driver'
    head,tail=text.split(marker,1)
    head=head.replace('  RETURN\n#endif',finish+'\n  RETURN\n#endif')
    source=out/'module_fddaobs_driver.F';source.write_text(head+marker+tail)
    flags=(BUILD/'CMakeFiles/WRF_Core.dir/flags.make').read_text()
    args=[]
    for key in ('Fortran_DEFINES','Fortran_INCLUDES','Fortran_FLAGS'):
        args+=shlex.split(re.search(rf'^{key} = (.*)$',flags,re.M)[1])
    args=[a for a in args if not a.startswith('-J')]
    compiler=Path('<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/gfortran')
    obj=out/'module_fddaobs_driver.F.o'
    command=[str(compiler),*args,'-J'+str(out),'-ffree-form','-cpp','-ffree-line-length-none','-fallow-argument-mismatch','-c',str(source),'-o',str(obj)]
    subprocess.run(command,cwd=out,check=True,timeout=180)
    link=shlex.split((BUILD/'main/CMakeFiles/wrf.dir/link.txt').read_text())
    link=[a for a in link if not a.startswith('-Wl,--dependency-file=')]
    link[0]=str(compiler)
    for i,a in enumerate(link):
        if a.endswith('.o') or a.endswith('.a'):link[i]=str((BUILD/'main'/a).resolve())
    where=link.index('-o');link[where+1]=str(out/'wrf_probe.exe')
    library=next(i for i,a in enumerate(link) if a.endswith('/libWRF_Core.a'))
    link.insert(library,str(obj))
    subprocess.run(link,cwd=out,check=True,timeout=180)
    (out/'PROVENANCE.json').write_text(json.dumps(dict(
        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        instrumented_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        executable_sha256=hashlib.sha256((out/'wrf_probe.exe').read_bytes()).hexdigest(),
        compiler=command,link=link),indent=2)+'\n')


if __name__=='__main__':main(Path(sys.argv[1]).resolve())
