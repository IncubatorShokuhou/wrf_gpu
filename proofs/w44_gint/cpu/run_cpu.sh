#!/usr/bin/env bash
# (w44-gint copy of w43-teacher run_cpu.sh: lane labels + expected wrfout count from CASE.json run_hours)
# One ORIGINAL CPU-WRF 4.7.1 72 h 3-nest run (lane w43-teacher): ERA5 reference or twin, from a case built by build_case.py.
# gen2 dmpar build (wrf sha 1d0f5779 = the WT2/WN3 CPU-reference executable), one MPI rank per listed PHYSICAL core
# (taskset per rank via OMPI_COMM_WORLD_RANK, nice 19), CPU-only OpenMPI env of FINDINGS E219 + per-rank libcuda check
# (any mapping -> abort), own session (deadline watchdog kills by SID: PRTE puts ranks in separate PGIDs, E31/D30), RAM cap
# via scripts/with_ram_slot.sh, sealed CASE/CPU_OUTPUTS.json + CASE/cpu_run.json. Opens nothing under an ALISIOS root.
# usage: run_cpu.sh CASE CORES(c0,c1,..) DEADLINE_ISO_UTC RAM_MAX(e.g. 12G|24G) TAG   (deadline = REAL UTC, real_utc.sh; CLOCK SKEW)
set -u
REAL=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
rnow() { local e; e=$("$REAL"); [[ $e =~ ^[0-9]+$ ]] && echo "$e" || date +%s; }
CASE=$(readlink -f "$1"); CORES=$2; DEADLINE=$(date -u -d "$3" +%s) || exit 2; RAM=$4; TAG=$5
BUILD=<DATA_ROOT>/canairy_meteo/artifacts/wrf_src/WRF/install_gen2_dmpar
WENV=<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build
EXE_SHA=1d0f5779fff143361e470e37f89c734497ed5a7bb55d0c53b7e3fccc569dd49b
TABLES=<DATA_ROOT>/wrf_gpu2/scratch/w41-icloud2/wn3_icloud2/cpu   # Thompson .dat written by this same executable
MBX=${MBX_OVERRIDE:-<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md}
TREE=<USER_HOME>/src/wrf_gpu2
RUN=$CASE/cpu
log(){ echo "$(date -u +%FT%TZ) [real $("$REAL" --iso)] $TAG $*" | tee -a "$CASE/run_cpu.log"; }
IFS=, read -r -a CPUS <<< "$CORES"; NR=${#CPUS[@]}
for c in "${CPUS[@]}"; do case ",0,16,6,22,14,15,30,31," in *",$c,"*) log "refusing core $c"; exit 2;; esac; done
NB=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['ranks'])" "$CASE/CASE.json") || exit 2
[ "$NB" = "$NR" ] || { log "case built for $NB ranks, $NR cores given"; exit 2; }
[ -e "$RUN" ] && { log "refusing existing $RUN (fresh case per attempt, E42)"; exit 2; }
for f in "$CASE"/input/*; do case "$(readlink -f "$f")" in <DATA_ROOT>/alisios/*|<DATA_ROOT>/server/work/*) log "ALISIOS-root input $f"; exit 2;; esac; done
[ "$(sha256sum "$BUILD/bin/wrf" | cut -c1-64)" = "$EXE_SHA" ] || { log "executable sha mismatch"; exit 2; }
while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 30; (( $(rnow) < DEADLINE - 3600 )) || { log "QUIET until deadline"; exit 3; }; done
FREE_G=$(df --output=avail -B1G <DATA_ROOT> | tail -1); (( FREE_G >= ${DISK_FLOOR_G:-365} )) || { log "refuse: <DATA_ROOT> free ${FREE_G} GiB < ${DISK_FLOOR_G:-365} (manager 15:45Z)"; exit 4; }
taskset -pc "$CORES" $$ > /dev/null
mkdir -p "$RUN"
for f in "$BUILD"/run/*; do ln -s "$f" "$RUN/$(basename "$f")"; done
rm -f "$RUN/namelist.input" "$RUN/wrf.exe" "$RUN"/wrfinput_d0* "$RUN"/wrfbdy_d01 "$RUN"/wrflowinp_d0*
cp "$CASE"/input/* "$RUN/"
for t in freezeH2O.dat qr_acr_qg_V4.dat qr_acr_qsV2.dat; do [ -s "$TABLES/$t" ] && cp "$TABLES/$t" "$RUN/"; done
cat > "$RUN/rank.sh" <<EOF
#!/bin/bash
CPUS=(${CPUS[*]})
exec taskset -c \${CPUS[\$OMPI_COMM_WORLD_RANK]} nice -n 19 $BUILD/bin/wrf
EOF
chmod +x "$RUN/rank.sh"
printf '#!/bin/bash\ndate +%%s.%%N > ../cpu_t0\necho $$ > ../cpu.sid\nexec "$@"\n' > "$RUN/sid_exec.sh"; chmod +x "$RUN/sid_exec.sh"
export LD_LIBRARY_PATH="$WENV/lib:${LD_LIBRARY_PATH:-}" PATH="$WENV/bin:$PATH" OMP_NUM_THREADS=1
export OMPI_MCA_opal_cuda_support=0 OMPI_MCA_accelerator=null OMPI_MCA_rcache=grdma OMPI_MCA_pml=ob1 \
       OMPI_MCA_btl=self,sm,tcp OMPI_MCA_coll=^ucc,hcoll UCX_TLS=self,sm,tcp HWLOC_COMPONENTS=-cuda,-nvml
BUDGET=$(( (DEADLINE - $(rnow) - 120) * 9 / 10 ))   # system-clock timeout x0.9 (slew); (( BUDGET > 3600 )) || { log "no time left ($BUDGET s)"; exit 4; }
log "launch $NR ranks on $CORES, budget ${BUDGET}s, RAM $RAM"
# watchdog: per-rank libcuda/affinity check once, then deadline kill by session id
( cd "$RUN"; checked=0
  while sleep 20; do
    [ -s ../cpu.sid ] || continue; SID=$(cat ../cpu.sid)
    ps -eo sid= | grep -qw "$SID" || exit 0
    if (( checked == 0 )); then
      mapfile -t R < <(ps -eo pid=,sid=,comm= | awk -v s="$SID" '$2==s && $3=="wrf"{print $1}')
      if (( ${#R[@]} == NR )); then
        checked=1; bad=0; : > ../rank_check.txt
        for p in "${R[@]}"; do n=$(grep -c libcuda /proc/$p/maps 2>/dev/null); d=$(ls -l /proc/$p/fd 2>/dev/null | grep -c /dev/nvidia)
          echo "pid $p cpus $(taskset -cp $p | awk -F': ' '{print $2}') libcuda_maps $n nvidia_fds $d" >> ../rank_check.txt
          (( n == 0 && d == 0 )) || bad=1; done
        echo "$(date -u +%FT%TZ) w44-gint: [real $("$REAL" --iso)] CPU START $TAG $NR ranks on $CORES nice19, libcuda $( awk '{print $6}' ../rank_check.txt | paste -sd+ ) nvidia-fds $( awk '{print $8}' ../rank_check.txt | paste -sd+ ), deadline $(date -u -d @$DEADLINE +%FT%TZ), run $RUN" >> "$MBX"
        if (( bad )); then echo "$(date -u +%FT%TZ) libcuda/nvidia mapped -> abort" >> ../watchdog.log
          for p in $(ps -eo pid=,sid= | awk -v s="$SID" '$2==s{print $1}'); do kill -TERM "$p"; done; fi
      fi
    fi
    if (( $(rnow) >= DEADLINE )); then echo "$(date -u +%FT%TZ) DEADLINE kill $SID" >> ../watchdog.log
      for p in $(ps -eo pid=,sid= | awk -v s="$SID" '$2==s{print $1}'); do kill -TERM "$p" 2>/dev/null; done; sleep 15
      for p in $(ps -eo pid=,sid= | awk -v s="$SID" '$2==s{print $1}'); do kill -KILL "$p" 2>/dev/null; done; exit 0; fi
  done ) &
WD=$!
cd "$RUN"
"$TREE/scripts/with_ram_slot.sh" --label "w44-gint-$TAG" --max "$RAM" --timeout 3600 -- \
  setsid ./sid_exec.sh timeout --signal=TERM --kill-after=30 "$BUDGET" \
  taskset -c "$CORES" nice -n 19 "$WENV/bin/mpirun" --use-hwthread-cpus --bind-to none -np "$NR" ./rank.sh > ../wrf.log 2>&1
RC=$?
kill "$WD" 2>/dev/null
T1=$(date +%s.%N); T0=$(cat ../cpu_t0 2>/dev/null || echo "$T1")
OK=0; grep -q "SUCCESS COMPLETE WRF" rsl.out.0000 2>/dev/null && OK=1
NR=$NR CORES=$CORES python3 - "$CASE" "$BUILD/bin/wrf" "$RC" "$T0" "$T1" "$OK" <<'EOF'
import hashlib, json, os, sys
from pathlib import Path
from netCDF4 import Dataset
case, exe, rc, t0, t1, ok = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6] == "1"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
rows = []
for f in sorted((case / "cpu").glob("wrfout_d0*")):
    with Dataset(f) as ds:
        times = [b"".join(t).decode() for t in ds["Times"][:]]
    rows.append({"file": f.name, "bytes": f.stat().st_size, "sha256": sha(f), "times": times})
(case / "CPU_OUTPUTS.json").write_text(json.dumps(rows, indent=1) + "\n")
rec = {"schema": "w44-gint.cpu_run.v1", "rc": rc, "success_complete": ok, "wall_s": t1 - t0, "wrfout_files": len(rows),
       "executable": exe, "executable_sha256": sha(exe), "ranks": int(os.environ["NR"]), "cores": os.environ["CORES"],
       "input_sha256": {p.name: sha(p) for p in sorted((case / "input").iterdir())},
       "rank_check": (case / "rank_check.txt").read_text() if (case / "rank_check.txt").exists() else None,
       "output_manifest_sha256": sha(case / "CPU_OUTPUTS.json")}
(case / "cpu_run.json").write_text(json.dumps(rec, indent=1) + "\n")
print(json.dumps({k: rec[k] for k in ("rc", "success_complete", "wall_s", "wrfout_files")}))
EOF
N=$(ls "$RUN"/wrfout_d0* 2>/dev/null | wc -l); NEXP=$(python3 -c "import json,sys;j=json.load(open(sys.argv[1]));print(j.get('max_dom',3)*(j.get('run_hours',72)+1))" "$CASE/CASE.json")
log "END rc=$RC success=$OK wrfout=$N/$NEXP"
echo "$(date -u +%FT%TZ) w44-gint: [real $("$REAL" --iso)] CPU END $TAG rc=$RC success=$OK wrfout=$N/$NEXP wall=$(python3 -c "print(round($T1-$T0))")s; cores $CORES free" >> "$MBX"
[ "$OK" = 1 ] && [ "$N" = "$NEXP" ]
