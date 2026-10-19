#!/bin/bash
# Launch ONE original CPU-WRF twin (lane w41-nestqual): one MPI rank per listed core (3 or 4; taskset per rank,
# nice 19), own session (killed by SID: PRTE puts ranks in separate PGIDs, D30/E31), hard deadline, ALISIOS run
# registry start/end (register.py, as alisios_mgr asked), seal on success. No ALISIOS path is opened here.
# usage: run_twin.sh <case_dir> <c0,c1,c2[,c3]> <deadline ISO-UTC> <live|dry> [hours=24] [ram_slot_max=10G|none]
#   live = also announces START (when WRF really starts, after the RAM slot) and END in the main MAILBOX.
#   dry = registry events go to a lane-private atlas dir (ALISIOS registry untouched), run_id gets a :dry suffix.
set -u
CASE=$(readlink -f "$1"); CORES=$2; DEADLINE=$(date -u -d "$3" +%s) || exit 2; MODE=$4; HOURS=${5:-24}; RAM=${6:-10G}
LANE=<USER_HOME>/wrf_gpu2_lanes/w41-nestqual; HERE=$(dirname "$(readlink -f "$0")")
MPI=<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/mpirun
REG=<USER_HOME>/src/alisios/ops/northstar/data_map/register.py
TREE=$(readlink -f "$HERE/../..")
log(){ echo "$(date -u +%FT%TZ) $(basename "$CASE") $*" | tee -a "$LANE/twin_status.log"; }
MAILBOX=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
IFS=, read -r -a CORE_ARR <<< "$CORES"; NR=${#CORE_ARR[@]}; (( NR == 3 || NR == 4 )) || { log "need 3 or 4 cores"; exit 2; }
for c in "${CORE_ARR[@]}"; do case ",14,15,30,31," in *",$c,"*) log "refusing OS core $c"; exit 2;; esac; done
taskset -pc "$CORES" $$ > /dev/null   # launcher, watchdog, timeout/prterun stay on the granted cores
cd "$CASE/cpu" || exit 2
[ -e cpu_run.json ] || [ -e ../cpu_run.json ] && { log "already sealed, refusing"; exit 2; }
ls wrfout_d0* > /dev/null 2>&1 && { log "wrfout present, refusing (fresh case dir per attempt, E42)"; exit 2; }
while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 30; (( $(date +%s) < DEADLINE - 600 )) || { log "QUIET until deadline, not started"; exit 3; }; done
ISSUE=$(python3 -c "import json;print(json.load(open('$CASE/PREPARE.json'))['issue'])")
ARM=$(python3 -c "import json;print(json.load(open('$CASE/PREPARE.json'))['arm'])")
PR=$(python3 -c "import json;print(json.load(open('$CASE/PREPARE.json'))['executable']['ranks'])")
[ "$PR" = "$NR" ] || { log "case was built for $PR ranks, $NR cores given (rebuild with --ranks $NR)"; exit 2; }
RID="wrf_gpu2-w41-nestqual:$ISSUE:$ARM:cpu_twin${HOURS}h${TWIN_ATTEMPT:+:$TWIN_ATTEMPT}"   # TWIN_ATTEMPT=a2.. for retries (registry ids are unique)
if [ "$MODE" = dry ]; then
  export ALISIOS_ATLAS_DIR=$LANE/registry_dry ALISIOS_DATA_MAP_NO_REGEN=1; mkdir -p "$ALISIOS_ATLAS_DIR"; RID="$RID:dry$(date -u +%H%M%S)"
elif [ "$MODE" != live ]; then log "mode must be live|dry"; exit 2; fi
cat > rank.sh <<EOF
#!/bin/bash
# one MPI rank per listed physical core (w41-nestqual run_twin.sh)
CORES=(${CORE_ARR[*]})
exec taskset -c \${CORES[\$OMPI_COMM_WORLD_RANK]} nice -n 19 ./wrf.exe
EOF
chmod +x rank.sh
STARTED=$(date -u +%FT%TZ)
PYTHONDONTWRITEBYTECODE=1 python3 -B "$REG" wrf start --run-id "$RID" --producer wrf_gpu2-w41-nestqual \
  --purpose "Original CPU-WRF 4.7.1 twin of G2 arm $ARM (nest gate D6 vs the G2 GPU run, v0.4.1); no product, no stations" \
  --family cpu_twin_v041_nestgate --tags "wrf_gpu2,w41-nestqual,g2_cpu_twin,$ARM,cpu_wrf_4.7.1_dmpar" --forcing AIFS \
  --engine "WRF V4.7.1 dmpar original CPU (wrf.exe sha 1d0f5779), $NR ranks" --nests "9/3/1 km" --lead-h "$(( HOURS > 0 ? HOURS : 1 ))" \
  --issue "$ISSUE" --run-dir "$CASE/cpu" --retention raw_delete_after_extract --started-utc "$STARTED" \
  --by wrf_gpu2-w41-nestqual > ../registry_start.log 2>&1 || { log "registry start FAILED ($RID), not started"; cat ../registry_start.log; exit 4; }
echo "$RID" > ../run_id; echo "$STARTED" > ../cpu_start_utc; echo "$CORES" > ../cpu_cores
BUDGET=$(( DEADLINE - $(date +%s) - 60 )); (( BUDGET > 300 )) || { log "no time left"; BUDGET=0; }
log "START $RID cores=$CORES budget=${BUDGET}s"
RAMWRAP=(); [ "$RAM" != none ] && RAMWRAP=("$TREE/scripts/with_ram_slot.sh" --label w41-nestqual --max "$RAM" --timeout 3600 --)
T0=$(date +%s.%N)
if (( BUDGET > 0 )); then
  # deadline watchdog: TERM then KILL every process of the run's session (ranks live in their own PGIDs)
  ( while sleep 20; do
      [ -s ../cpu.sid ] || continue; SID=$(cat ../cpu.sid)
      if [ "$MODE" = live ] && [ ! -e ../mailbox_start ]; then
        touch ../mailbox_start
        echo "$(date -u +%FT%TZ) w41-nestqual: CPU TWIN START $ARM ($RID) cores $CORES, $NR ranks nice19, deadline $(date -u -d @$DEADLINE +%FT%TZ), est ~3 h" >> "$MAILBOX"
      fi
      ps -eo sid= | grep -qw "$SID" || exit 0
      if (( $(date +%s) >= DEADLINE )); then
        echo "$(date -u +%FT%TZ) DEADLINE kill session $SID" >> ../cpu.log
        for p in $(ps -eo pid=,sid= | awk -v s="$SID" '$2==s{print $1}'); do kill -TERM "$p" 2>/dev/null; done; sleep 15
        for p in $(ps -eo pid=,sid= | awk -v s="$SID" '$2==s{print $1}'); do kill -KILL "$p" 2>/dev/null; done; exit 0
      fi
    done ) &
  WD=$!
  # no '$' in argv: systemd-run (inside with_ram_slot) expands $VAR/$$ in its command line
  printf '#!/bin/bash\ndate +%%s.%%N > ../cpu_t0\necho $$ > ../cpu.sid\nexec "$@"\n' > sid_exec.sh; chmod +x sid_exec.sh
  "${RAMWRAP[@]}" setsid ./sid_exec.sh timeout --signal=TERM --kill-after=30 "$BUDGET" \
    taskset -c "$CORES" nice -n 19 "$MPI" --use-hwthread-cpus --bind-to none -np "$NR" ./rank.sh > ../cpu.log 2>&1
  RC=$?
  kill "$WD" 2>/dev/null
else RC=124; fi
T1=$(date +%s.%N); ENDED=$(date -u +%FT%TZ); [ -s ../cpu_t0 ] && T0=$(cat ../cpu_t0)   # wall excludes the RAM-slot wait
python3 - "$T0" "$T1" "$RC" > ../cpu_wall.json <<'EOF'
import json, sys; t0, t1, rc = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
print(json.dumps(dict(t0_epoch=t0, t1_epoch=t1, wall_s=t1 - t0, rc=rc)))
EOF
STATUS=FAILED
if [ "$RC" = 0 ] && grep -q "SUCCESS COMPLETE WRF" rsl.out.0000 2>/dev/null; then
  if JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 "$HERE/seal_twin.py" "$CASE" "$HOURS" > ../seal.log 2>&1; then STATUS=DONE_CLEAN
  else STATUS=DONE_HELD; fi
elif [ "$RC" = 124 ] || [ "$RC" = 137 ] || [ "$RC" = 143 ]; then STATUS=ABORTED; fi
PYTHONDONTWRITEBYTECODE=1 python3 -B "$REG" wrf end --run-id "$RID" --ended-utc "$ENDED" --rc "$RC" --status "$STATUS" \
  --corpus none --raw-deleted nein --notes "wrf_gpu2 w41-nestqual CPU twin; raw wrfout kept in wrf_gpu2 scratch until the D6 critic verdict, then deleted" \
  --by wrf_gpu2-w41-nestqual > ../registry_end.log 2>&1 || log "registry end FAILED ($RID)"
log "END $RID rc=$RC status=$STATUS frames=$(ls wrfout_d0* 2>/dev/null | wc -l) wall=$(python3 -c "print(round($T1-$T0))")s"
[ "$MODE" = live ] && echo "$(date -u +%FT%TZ) w41-nestqual: CPU TWIN END $ARM rc=$RC status=$STATUS frames=$(ls wrfout_d0* 2>/dev/null | wc -l) wall=$(python3 -c "print(round($T1-$T0))")s; cores $CORES free" >> "$MAILBOX"
[ "$STATUS" = DONE_CLEAN ]
