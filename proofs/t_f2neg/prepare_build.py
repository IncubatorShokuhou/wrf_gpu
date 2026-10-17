"""Stage print-only instrumentation in an existing private pristine build."""
import hashlib,json,pathlib
ROOT=pathlib.Path('<USER_HOME>/wrf_gpu2_lanes/t-f2neg')
BUILD=ROOT/'build'
original=pathlib.Path('<USER_HOME>/src/wrf_pristine/WRF/dyn_em/solve_em.f90')
source=original.read_text()
for name in ('em_quarter_ss_center_dump.txt','em_grav2d_x_touchdown_dump.txt'):
    source=source.replace('<DATA_ROOT>/wrf_gpu2/wrf_truth/'+name,str(ROOT/name))
fields=[('moist','qv'),('moist','qc'),('moist','qr'),('moist','qi'),('moist','qs'),('moist','qg'),('scalar','qni'),('scalar','qnr'),('scalar','qnc'),('scalar','qnn')]
calls=[]
for fid,(group,name) in enumerate(fields,1):
    calls.append(f'''   IF (P_{name}>1 .and. P_{name}<=num_{group}) &
     CALL f2_observe({group}(:,:,:,P_{name}),{fid},grid%itimestep, &
       ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme)''')
needle='   END DO Runge_Kutta_loop'
assert source.count(needle)==1
source=source.replace(needle,needle+'\n'+'\n'.join(calls))
(BUILD/'dyn_em/solve_em.f90').write_text(source)
observer=pathlib.Path(__file__).with_name('observer.f90').read_text()
(BUILD/'dyn_em/f2_observer.f90').write_text(observer)
make=f'''WRF_SRC_ROOT_DIR={BUILD}
include {BUILD}/configure.wrf
all:
\t$(SFC) $(FCFLAGS) $(INCLUDE_MODULES) -I{BUILD}/dyn_em -c {BUILD}/dyn_em/solve_em.f90 -o {BUILD}/dyn_em/solve_em.o
\t$(SFC) $(FCFLAGS) -c {BUILD}/dyn_em/f2_observer.f90 -o {BUILD}/dyn_em/f2_observer.o
\t$(AR) r {BUILD}/main/libwrflib.a {BUILD}/dyn_em/solve_em.o {BUILD}/dyn_em/f2_observer.o
\t$(SFC) -o {ROOT}/wrf_observed.exe $(LDFLAGS) {BUILD}/main/wrf.o {BUILD}/main/module_wrf_top.o {BUILD}/main/libwrflib.a $(LIB)
'''
(ROOT/'observer.mk').write_text(make)
sha=lambda b:hashlib.sha256(b).hexdigest()
(ROOT/'BUILD_SOURCE.json').write_text(json.dumps(dict(original_f90=str(original), original_sha256=sha(original.read_bytes()),instrumented_sha256=sha(source.encode()),observer_sha256=sha(observer.encode()),fields=fields,scope='only read-only calls after END DO Runge_Kutta_loop plus private legacy diagnostic paths'),indent=2)+'\n')
