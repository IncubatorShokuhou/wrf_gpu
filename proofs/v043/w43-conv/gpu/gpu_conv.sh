#!/usr/bin/env bash
# w43-conv GPU arm CG01-CG08 (IMMUTABLE once its sha is posted): the CPU-validated same-input pristine oracles re-run with
# the TRITON-COMPILED product kernels on the asserted GPU (CPU interpret != GPU compile, E17/E84):
#   CG01 Thompson full_column on 586 REAL convective #148 columns vs CT02 pristine mp_gt_driver outputs;
#   CG02 sixth-order diffusion (diff_6th_opt 2) on 11 whole convective frames vs pristine sixth_order_diffusion;
#   CG03 MYNN public call on cold-pool/core/downdraft/env columns vs pristine mynnedmf_driver;
#   CG04 PD scalar advection (native Pallas) vs pristine advect_scalar_pd; CG05 MYNN-SL; CG06 km_opt=4 Kh chain;
#   CG07 horizontal diffusion with convective Kh (XLA/adapter paths on GPU); CG08 acoustic small step (Triton) on the
#   CPU-built d06 convective stage fixture (analysis/data/cw_fixture_d06.npz) vs the same pristine routines.
# Window (manager slot, REAL UTC via <USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh, never raw date; clock-skew rule
# 2026-10-20): START 2026-10-21T11:35:00Z, LOCK_BY 11:45:00Z, END 12:05:00Z. Lease /tmp/wrf_gpu2_run_approvals/w43-conv.lease.
# Inside the lock REM = END - real_now - 60 s; refuse < 1200 s (rc 6); the python chain runs under timeout 0.9*REM system
# seconds (chrony slew margin); lock wait = min(600, LOCK_BY - real_now) * 0.9. Both clocks recorded in clocks.txt.
# GPUWRF_JAX_CACHE=0, no prealloc, private autotune dir; hibernation journal before/after.
# IMMUTABLE SNAPSHOT (critic delta 1): W is a detached git worktree of lane/w43-conv at the frozen commit (+ data/wrf_pristine
# link, + copied untracked inputs analysis/data/{frames,conv_columns,ct02_oracle,cw_fixture_d06.npz(.json)}); the waiter runs
# THIS file from the snapshot. Refuses (rc 13) unless HEAD is detached and the running file == the committed blob.
# FROZEN MEASURED SOURCE (critic cr-w43-conv): before the lock the launcher refuses unless (rc 10) every script, helper,
# pristine source/driver and input/reference file matches analysis/gpu/FROZEN_MANIFEST.sha256 (sha256sum -c), (rc 11) the
# worktree src tree hash == b3e9b92a8ce2f1aa4a96f39b0b03424a3c1ebb42 (= v0.4.2 src tree), (rc 12) src/ tests/ proofs/
# scripts/ analysis/*.py have no uncommitted change. Setup probes (nvidia-smi, journalctl) run under timeout 20.
# Verdicts = per-arm numeric gates (analysis/gpu/gpu_verdicts.py -> VERDICTS.json), never launcher/process rc.
set -u
W=<DATA_ROOT>/wrf_gpu2/scratch/w43-conv/gpu_snapshot   # DETACHED immutable snapshot (AGENTS: measure snapshots, never a live checkout)
M=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
LOCK=<USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sh
RU=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
START=$(date -u -d "2026-10-21T11:35:00Z" +%s)
LOCK_BY=$(date -u -d "2026-10-21T11:45:00Z" +%s)
END=$(date -u -d "2026-10-21T12:05:00Z" +%s)
OUTD=$W/analysis/data/gpu_conv
cd "$W" || exit 7
now=$("$RU"); case "$now" in ''|*[!0-9]*) echo "real_utc.sh failed"; exit 7;; esac
if [ "$now" -lt "$START" ]; then echo "too early (real $("$RU" --iso))"; exit 4; fi
WAIT=$(( (LOCK_BY - now) * 9 / 10 )); [ "$WAIT" -gt 540 ] && WAIT=540
if [ "$WAIT" -lt 10 ]; then echo "lock deadline 11:45Z passed (real $("$RU" --iso))"; exit 6; fi
[ -e /tmp/wrf_gpu2_run_approvals/w43-conv.lease ] || { echo "no lease"; exit 5; }
git -C "$W" symbolic-ref -q HEAD > /dev/null && { echo "snapshot not detached"; exit 13; }
[ "$(realpath "$0")" = "$W/analysis/gpu/gpu_conv.sh" ] || { echo "not running from the snapshot"; exit 13; }
[ "$(git -C "$W" hash-object "$0")" = "$(git -C "$W" rev-parse HEAD:analysis/gpu/gpu_conv.sh)" ] || { echo "launcher != committed blob"; exit 13; }
sha256sum -c --quiet analysis/gpu/FROZEN_MANIFEST.sha256 > /dev/null 2>&1 || { echo "manifest mismatch"; exit 10; }
[ "$(git -C "$W" rev-parse HEAD:src)" = b3e9b92a8ce2f1aa4a96f39b0b03424a3c1ebb42 ] || { echo "src tree mismatch"; exit 11; }
[ -z "$(git -C "$W" status --porcelain --untracked-files=no)" ] || { echo "dirty tree"; exit 12; }
mkdir -p "$OUTD" || exit 7
[ -e "$OUTD/DONE" ] && { echo "already done"; exit 9; }
timeout 20 journalctl -k --since "-2h" 2>/dev/null | grep -i hibernat > "$OUTD/journal_before.txt"
SHA=$(sha256sum analysis/gpu/FROZEN_MANIFEST.sha256 | cut -d" " -f1)
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) w43-conv: [real $("$RU" --iso)] GPU START CG01-08 convective pristine oracles on Triton kernels (release defaults, manifest sha256 $SHA, src tree b3e9b92a8c, lock wait <= ${WAIT}s, real END 12:05Z)" >> "$M"
export W OUTD RU END
"$LOCK" --label w43-conv --timeout "$WAIT" -- bash -c '
  set -u
  now=$("$RU"); REM=$(( END - now - 60 ))
  { echo "real $("$RU" --iso) sys $(date -u +%FT%TZ) REM=$REM"; } >> "$OUTD/clocks.txt"
  if [ "$REM" -lt 1200 ]; then echo "inside lock: window too short (${REM}s)"; exit 6; fi
  TO=$(( REM * 9 / 10 ))
  { date -u +%FT%TZ; timeout 20 nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv;
    timeout 20 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; } > "$OUTD/nvidia_smi_start.txt" 2>&1
  E="env -u JAX_PLATFORMS XLA_PYTHON_CLIENT_PREALLOCATE=false GPUWRF_JAX_CACHE=0 GPUWRF_XLA_AUTOTUNE_CACHE_DIR=$OUTD/autotune PYTHONPATH=$W/src W43_PLATFORM=gpu"
  timeout -k 15 "$TO" bash -c "
    $E python -u $W/analysis/gpu/thompson_gpu.py $W/analysis/data/conv_columns.npz $W/analysis/data/ct02_oracle.npz $OUTD/cg01_thompson_gpu.npz > $OUTD/cg01.log 2>&1; echo CG01 rc \$? >> $OUTD/rc.txt
    $E python -u $W/analysis/dyn/sixth_oracle.py $W/analysis/data/frames/*.npz --out $OUTD/cg02_sixth_gpu.json > $OUTD/cg02.log 2>&1; echo CG02 rc \$? >> $OUTD/rc.txt
    $E python -u $W/analysis/mynn/mynn_oracle.py $W/analysis/data/frames $OUTD/cg03_mynn_gpu.json > $OUTD/cg03.log 2>&1; echo CG03 rc \$? >> $OUTD/rc.txt
    $E python -u $W/analysis/dyn/pd_oracle.py $W/analysis/data/frames/*.npz --out $OUTD/cg04_pd_gpu.json > $OUTD/cg04.log 2>&1; echo CG04 rc \$? >> $OUTD/rc.txt
    $E python -u $W/analysis/mynn/mynnsl_oracle.py $W/analysis/data/frames $OUTD/cg05_mynnsl_gpu.json > $OUTD/cg05.log 2>&1; echo CG05 rc \$? >> $OUTD/rc.txt
    $E python -u $W/analysis/dyn/kh_oracle.py $W/analysis/data/frames/*.npz --out $OUTD/cg06_kh_gpu.json > $OUTD/cg06.log 2>&1; echo CG06 rc \$? >> $OUTD/rc.txt
    $E python -u $W/analysis/dyn/hdiff_oracle.py $W/analysis/data/frames/*.npz --out $OUTD/cg07_hdiff_gpu.json > $OUTD/cg07.log 2>&1; echo CG07 rc \$? >> $OUTD/rc.txt
    $E python -u $W/analysis/dyn/acoustic_oracle.py --from-fixture $W/analysis/data/cw_fixture_d06.npz $OUTD/cg08_acoustic_gpu.json > $OUTD/cg08.log 2>&1; echo CG08 rc \$? >> $OUTD/rc.txt
  "
'
rc=$?
VER=$(JAX_PLATFORMS=cpu timeout 600 taskset -c 8,9,12 nice -n 19 python3 "$W/analysis/gpu/gpu_verdicts.py" "$OUTD" 2>&1 | tail -1)
timeout 20 journalctl -k --since "-2h" 2>/dev/null | grep -i hibernat > "$OUTD/journal_after.txt"
touch "$OUTD/DONE"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) w43-conv: [real $("$RU" --iso)] GPU END lock_rc=$rc NUMERIC VERDICTS: $VER | arm rc: $(tr '\n' ' ' < $OUTD/rc.txt 2>/dev/null) (outputs $OUTD; scoring on CPU next)" >> "$M"
exit $rc
