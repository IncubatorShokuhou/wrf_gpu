#!/bin/bash
# Q-B: CPU-WRF (original ALISIOS dmpar exe sha 1d0f5779) max_dom 3 vs 9, same inputs/namelist (W9N 20260316_12z input42,
# guarded stage S02_w9n0316), only max_dom differs; run_hours=$HOURS, decomposition 1x3 in both, ranks pinned 8,9,12 nice 19.
# usage: run_maxdom.sh <n3|n9> HOURS
set -u
ARM=$1; HOURS=$2
STAGE=<DATA_ROOT>/wrf_gpu2/scratch/w43-coastwind/stage/S02_w9n0316/root2/scratch/w1011-ds-coast-wrf-value/stage2/cases/20260316_12z/input42
RUN=<DATA_ROOT>/wrf_gpu2/worktrees/w43-coastwind/analysis/maxdom
EXE_DIR=<DATA_ROOT>/canairy_meteo/artifacts/wrf_src/WRF/install_gen2_dmpar/run
MPIRUN=<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/mpirun
D=<DATA_ROOT>/wrf_gpu2/scratch/w43-coastwind/maxdom/$ARM
[ -e "$D" ] && { echo "exists $D"; exit 2; }
mkdir -p "$D"; cd "$D"
for f in "$EXE_DIR"/*; do b=$(basename "$f"); case $b in namelist.input|wrfinput_*|wrfbdy_*|wrflowinp_*|wrfout*|rsl.*) ;; *) ln -s "$f" "$b";; esac; done
ND=${ARM:1:1}; [[ $ND =~ ^[39]$ ]] || { echo "arm must be n3*|n9*"; exit 2; }
for f in wrfbdy_d01 $(cd "$STAGE"; ls wrfinput_d0* wrflowinp_d0* | awk -v n=$ND '{d=substr($0,length($0),1)+0; if (d<=n) print}'); do ln -s "$STAGE/$f" "$f"; done
python3 - "$STAGE/namelist.input" namelist.input "$ND" "$HOURS" <<'PY'
import re,sys
src,dst,nd,h=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4])
t=open(src).read()
def sub(key,val):
    global t
    t,n=re.subn(r'(?m)^(\s*%s\s*=).*$'%key, r'\g<1> %s,'%val, t); assert n==1,key
sub('max_dom',nd); sub('run_hours',h); sub('nproc_x',1); sub('nproc_y',3); sub('restart_interval',100000)
# end time = start + h (start 2026-03-16_18)
import datetime as dt
e=dt.datetime(2026,3,16,18)+dt.timedelta(hours=h)
for k,v in (('end_month',e.month),('end_day',e.day),('end_hour',e.hour)):
    sub(k,', '.join([str(v)]*9))
open(dst,'w').write(t)
PY
diff "$STAGE/namelist.input" namelist.input > namelist.diff
cat > rank.sh <<'R'
#!/bin/bash
CORES=(8 9 12)
exec taskset -c ${CORES[$OMPI_COMM_WORLD_RANK]} nice -n 19 ./wrf.exe
R
chmod +x rank.sh
export OMPI_MCA_opal_cuda_support=0 OMPI_MCA_accelerator=null OMPI_MCA_rcache=grdma OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,sm,tcp OMPI_MCA_coll=^ucc,hcoll UCX_TLS=self,sm,tcp HWLOC_COMPONENTS=-cuda,-nvml CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1
date -u +%FT%TZ > start_utc
( journalctl -k --since "-5min" 2>/dev/null | grep -i hibernat ) > hib_before.txt
setsid bash -c "echo \$\$ > pgid; exec timeout -k 30 7200 $MPIRUN --oversubscribe -np 3 --bind-to none ./rank.sh" > mpirun.log 2>&1 &
sleep 20
for p in $(pgrep -f "$D/wrf.exe" ; pgrep -x wrf.exe); do [ "$(readlink -f /proc/$p/cwd)" = "$D" ] && echo "$p libcuda=$(grep -c libcuda /proc/$p/maps) nvfds=$(ls -l /proc/$p/fd 2>/dev/null | grep -c nvidia) cpus=$(taskset -cp $p | awk -F': ' '{print $2}')"; done | sort -u > ranks_check.txt
cat ranks_check.txt
wait
echo "rc=$?" > done.txt; date -u +%FT%TZ >> done.txt
( journalctl -k --since "$(cat start_utc | sed 's/T/ /;s/Z//')" 2>/dev/null | grep -i hibernat ) > hib_during.txt
tail -2 rsl.out.0000 >> done.txt
