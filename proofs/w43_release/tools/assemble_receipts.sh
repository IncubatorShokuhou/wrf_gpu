#!/usr/bin/env bash
# w43-release: copy the small receipts of the sealed v0.4.3 bundle session (v0.4.2 layout) — CPU, copies only.
#   P/receipts/<arm>/   every GPU-session arm: small files + proof/ (no wrfout/, scratch/, vram_*.csv), scope.log
#   P/meta/             BUNDLE_USAGE.md, session.log (GPU session), rw3t_session.log
#   P/e41/icloud0/      the icloud=0 E41 scan (default scan is already P/e41)
#   LANE_DIR            verdict-level receipts + BIG_FILES_SHA256SUMS for score files kept in $L (too big for git)
# usage: bash assemble_receipts.sh LANE_DIR       (LANE_DIR must not exist; P/receipts must not exist)
set -euo pipefail
P=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3
S=<DATA_ROOT>/wrf_gpu2/scratch/w43-release
L=<USER_HOME>/wrf_gpu2_lanes/w43-release
F=$L/followup
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$1
[ ! -e "$OUT" ] && [ ! -e "$P/receipts" ] || { echo "refusing: $OUT or $P/receipts exists"; exit 2; }
mkdir -p "$OUT" "$P/receipts" "$P/e41/icloud0"
small=(check_after.json check_after.txt check_before.json check_before.txt device.err device.txt env.txt
       foreign_flags.txt mtimes.json rc.txt run.log source_head.txt t0_unix t_end_unix time.txt wrfout_sha.txt)
for d in "$S"/gpu/runs/*/; do
  a=$(basename "$d"); mkdir -p "$P/receipts/$a/proof"
  for f in "${small[@]}"; do [ -e "$d/$f" ] && cp -p "$d/$f" "$P/receipts/$a/$f"; done
  cp -p "$d"/proof/* "$P/receipts/$a/proof/"
  cp -p "$S/gpu/runs/$a.scope.log" "$P/receipts/$a/scope.log"
done
cp -p "$HERE/../BUNDLE_USAGE.md" "$P/meta/BUNDLE_USAGE.md"
cp -p "$S/gpu/session.log" "$P/meta/session.log"
cp -p "$S/rw3t/session.log" "$P/meta/rw3t_session.log"
cp -p "$F"/e41_icloud0/* "$P/e41/icloud0/"
# lane receipts (small)
for f in RW_VERDICT.json RW_VERDICT_run1.json RW_TABLE_FINAL.json INTEGRITY_ADJUDICATION.json QGRAUP_D03_CENSUS.json \
         AUTOTUNE_DIFF_v043_vs_v042.json RW3T_VERDICT.json RW3T_ACCEPT.json RW3T_BYTES.json RW3T_DETAIL24.json \
         RW3T_DETAIL72.json RW3T_DISK.json FLOOR72_N72.json check_after_e41.json check_final.json followup.log \
         followup_run1.log; do cp -p "$F/$f" "$OUT/$f"; done
mkdir -p "$OUT/d6_N72" "$OUT/icloud0_score" "$OUT/aotkey" "$OUT/e41"
cp -p "$F"/d6_N72/summary.json "$F"/d6_N72/resources.json "$F"/d6_N72/*.md "$F"/d6_N72/*.log "$OUT/d6_N72/"
cp -p "$F"/icloud0_score/{D6_SUMMARY.json,ICLOUD0_report.json,cpu_run.json,CPU_OUTPUTS.json} "$OUT/icloud0_score/"
cp -p "$L"/aotkey_b979188128e0/*.json "$L"/aotkey_x_b979188128e0/*.json "$OUT/aotkey/"
cp -p "$P/e41/wave_gate.json" "$OUT/e41/wave_gate_default.json"
cp -p "$F/e41_icloud0/wave_gate.json" "$OUT/e41/wave_gate_icloud0.json"
cp -p "$L/VARDIFF_N1_vs_O1.json" "$OUT/VARDIFF_N1_vs_O1.json"
cp -p "$P/sealed/BUNDLE.json" "$OUT/BUNDLE.json"
cp -p "$S/gpu/session.log" "$OUT/gpu_session.log"
cp -p "$S/rw3t/session.log" "$OUT/rw3t_session.log"
(cd "$L" && sha256sum followup/d6_N72/d0?.json followup/d6_N72/d0?_early_hours.json followup/icloud0_score/D6_d0?.json \
   followup/icloud0_score/compare_d0?.json) > "$OUT/BIG_FILES_SHA256SUMS"
(cd "$OUT" && find . -type f ! -name SHA256SUMS -printf '%P\n' | LC_ALL=C sort | xargs -d '\n' sha256sum) > "$OUT/SHA256SUMS"
echo "assembled: $(find "$P/receipts" -type f | wc -l) bundle receipt files, $(find "$OUT" -type f | wc -l) lane files"
