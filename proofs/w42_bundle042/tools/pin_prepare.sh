#!/usr/bin/env bash
# w42-bundle042: CPU work right after the manager posts 'BUNDLE PIN <sha>' (no GPU, no JAX in this shell).
# usage: [RELID_CORES=c] [AOTKEY_CORES=c] [SKIP_RC=1] bash pin_prepare.sh PIN [ICLOUD0=auto|0|1] [CORES=8,9,12] [STAGE=all|jobs]
#        (ICLOUD0 auto: 1 iff the pin binds icloud=0)
#  1. pin checks: main:src == pin src; default SWDDIR fix present; icloud=0 binding (w42-icloud0 merged) decides ICLOUD0=auto
#  2. freeze the session tools: F=$L/tools_session_<pin12> = exact copy of the COMMITTED lane tools dir (clean required)
#  3. setup.sh F PIN ICLOUD0 (P/src worktree of PIN, bundle.env, P/tools, cases, pin.env, PREREG.json, SHA256SUMS)
#  4. PIN snapshot for the CPU checks: $S/snap_<pin12> (detached worktree + data/wrf_pristine link)
#  5. arm the GPU trigger from F (FROM = current MAILBOX line count) -> $L/trigger_<pin12>.log
#  6. detached CPU jobs, each in its own RAM slot on CORES: release identity (expect Swiss b977712f / PROD 321aa651),
#     aot-key pairs (F/aotkey_cpu.sh on the snapshot), RC delta fe4cda9cc9..PIN (rc_delta -> plan -> 2 workers).
#     Refused (exit 5, nothing launched) while a foreign process is PINNED to CORES (lent cores, manager 09:06:12Z):
#     ask the manager for the cores, then re-run with STAGE=jobs (steps 1-5 are not repeated).
set -euo pipefail
PIN=$1 ICLOUD0=${2:-auto} CORES=${3:-8,9,12} STAGE=${4:-all}
REPO=<USER_HOME>/src/wrf_gpu2; L=<USER_HOME>/wrf_gpu2_lanes/w42-bundle042; S=<DATA_ROOT>/wrf_gpu2/scratch/w42-bundle042
LANE=<DATA_ROOT>/wrf_gpu2/worktrees/w42-bundle042; MB=$REPO/.agent/kernel/MAILBOX.md
if [ -e /tmp/wrf_gpu2_quiet ]; then echo "QUIET: $(cat /tmp/wrf_gpu2_quiet)"; exit 3; fi
PIN=$(git -C "$REPO" rev-parse "$PIN^{commit}"); P12=${PIN:0:12}; TREE=$(git -C "$REPO" rev-parse "$PIN:src")
F=$L/tools_session_$P12; SNAP=$S/snap_$P12
if [ "$STAGE" = all ]; then
  [ "$(git -C "$REPO" rev-parse main:src)" = "$TREE" ] || { echo "main:src != pin src"; exit 4; }
  # no `git show | grep -q` under pipefail: grep -q exits at the first match, git gets SIGPIPE, the pipeline "fails"
  # (2026-10-19 11:11Z re-pin: icloud0 arms silently OFF on a pin that binds icloud=0) -> count matches instead
  NB=$(git -C "$REPO" show "$PIN:src/gpuwrf/io/namelist_binding.py" | /usr/bin/grep -c "icloud not in (0, 2, 3)" || true)
  if [ "${NB:-0}" -ge 1 ]; then BINDS0=1; else BINDS0=0; fi
  [ "$ICLOUD0" = auto ] && ICLOUD0=$BINDS0
  [ "$ICLOUD0" = 0 ] || [ "$BINDS0" = 1 ] || { echo "ICLOUD0=1 but the pin refuses icloud=0 (w42-icloud0 not merged)"; exit 4; }
  echo "icloud=0 bound by the pin: $BINDS0 -> icloud0 arms: $ICLOUD0"
  [ "$(git -C "$REPO" show "$PIN:src/gpuwrf/kernels/rad_sw_band_sums.py" | /usr/bin/grep -c 'dn \* bn')" -ge 2 ] \
    || { echo "pin lacks the default SWDDIR fix (dn * bn twice)"; exit 4; }
  [ -z "$(git -C "$LANE" status --porcelain -- proofs/w42_bundle042/tools)" ] || { echo "lane tools dirty: commit first"; exit 4; }
  [ ! -e "$F" ] || { echo "$F exists"; exit 4; }
  cp -a "$LANE/proofs/w42_bundle042/tools" "$F"
  echo "frozen from lane commit $(git -C "$LANE" rev-parse HEAD) at $(date -u +%FT%TZ)" > "$F/FROZEN_FROM.txt"
  bash "$F/setup.sh" "$F" "$PIN" "$ICLOUD0" | tee "$L/setup_$P12.log"
  [ -e "$SNAP" ] || git -C "$REPO" worktree add --detach "$SNAP" "$PIN" > /dev/null
  [ -e "$SNAP/data/wrf_pristine" ] || ln -s <USER_HOME>/src/wrf_pristine "$SNAP/data/wrf_pristine"
  [ "$(git -C "$SNAP" rev-parse HEAD:src)" = "$TREE" ] && [ -z "$(git -C "$SNAP" status --porcelain -- src)" ]
  FROM=$(wc -l < "$MB")
  setsid bash -c "exec bash '$F/trigger.sh' $FROM" > "$L/trigger_$P12.log" 2>&1 < /dev/null &
  echo "trigger armed from MAILBOX line $FROM (pid $!), log $L/trigger_$P12.log"
fi
[ -s "$F/pin.env" ] && [ -e "$SNAP/src" ] || { echo "run STAGE=all first"; exit 4; }
# lent cores (manager 09:06:12Z): a foreign process explicitly placed on lane cores (affinity a subset of the lane cores
# 8,9,12,13,24,25,28,29 that meets CORES) and using > 20 % of a core over 2 s = the lender still works there
python3 - "$CORES" <<'PY' || { echo "CORES $CORES BUSY (pinned foreign jobs above): ask the manager, then: bash $0 $PIN $ICLOUD0 $CORES jobs"; exit 5; }
import os, sys, time
cores, lane = {int(c) for c in sys.argv[1].split(",")}, {8, 9, 12, 13, 24, 25, 28, 29}
def cpu(pid):
    total = 0
    for tid in os.listdir(f"/proc/{pid}/task"):
        f = open(f"/proc/{pid}/task/{tid}/stat").read().rsplit(")", 1)[1].split()
        total += int(f[11]) + int(f[12])
    return total
cand = {}
for pid in filter(str.isdigit, os.listdir("/proc")):
    try:
        allowed = os.sched_getaffinity(int(pid))
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()
        if allowed <= lane and allowed & cores and cmd and "w42-bundle042" not in cmd:
            cand[pid] = (sorted(allowed), cmd[:160], cpu(pid))
    except (OSError, ValueError, IndexError):
        continue
time.sleep(2.0)
busy = []
for pid, (allowed, cmd, t0) in cand.items():
    try:
        share = (cpu(pid) - t0) / os.sysconf("SC_CLK_TCK") / 2.0
    except (OSError, ValueError, IndexError):
        continue
    if share > 0.20:
        busy.append((int(pid), allowed, round(share, 2), cmd))
for b in busy:
    print("PINNED-BUSY", *b)
sys.exit(1 if busy else 0)
PY
mkdir -p "$L/pin_$P12"
job() {  # job NAME CORES MAX CMD... : detached, own RAM slot, given cores, nice 19, CPU-only JAX
  local n=$1 c=$2 max=$3; shift 3
  setsid bash -c "exec '$LANE/scripts/with_ram_slot.sh' --label 'w42-bundle042-$n' --max '$max' --timeout 21600 -- \
    env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONDONTWRITEBYTECODE=1 taskset -c '$c' nice -n 19 $*" \
    > "$L/pin_$P12/$n.log" 2>&1 < /dev/null &
  echo "launched $n on $c (pid $!)"
}
[ "${SKIP_RELID:-0}" = 1 ] || job relid "${RELID_CORES:-$CORES}" 12G python3 "$SNAP/proofs/s1_port/release_identity.py" "$SNAP" "$L/pin_$P12/release_identity.json"
[ "${SKIP_RELID:-0}" = 1 ] || job relidprod "${RELID_CORES:-$CORES}" 12G python3 "$SNAP/proofs/s1_port/release_identity.py" "$SNAP" \
  "$L/pin_$P12/release_identity_prod.json" <DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725
job aotkey "${AOTKEY_CORES:-$CORES}" 12G bash "$F/aotkey_cpu.sh" "$SNAP"
[ "${SKIP_RC:-0}" = 1 ] && { echo "PIN_PREPARED (RC delta launched separately) pin=$PIN src=$TREE F=$F snap=$SNAP"; exit 0; }
python3 "$F/rc_delta.py" "$REPO" fe4cda9cc9a78aa5683173b2732f920ef2e99988 "$PIN" "$L/pin_$P12/rc_delta_files.txt"
D=$L/rc/delta_$P12
python3 "$F/rc_suite.py" plan --tree "$SNAP" --out "$D" --file-list "$L/pin_$P12/rc_delta_files.txt" \
  --baseline <USER_HOME>/wrf_gpu2_lanes/w41-knownred/FULL_FILE_RESULTS.json
bash "$F/rc_launch.sh" "$SNAP" "$D" "dA:${CORES%%,*}:6G:6G" "dB:${CORES##*,}:6G:6G"
echo "PIN_PREPARED pin=$PIN src=$TREE F=$F snap=$SNAP"
