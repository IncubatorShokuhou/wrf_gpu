#!/usr/bin/env bash
# w43-release: copy the small receipts of the sealed v0.4.3 bundle into P (v0.4.2 layout) and write P/SHA256SUMS over
# the same scope as the v0.4.2 bundle (bundle.env e41 meta receipts sealed tools; the cache is pinned by
# sealed/BUNDLE_MANIFEST.json). CPU, copies only; never touches P/jax, P/src or any wrfout.
#   P/receipts/<arm>/   every phase-1 arm: small files + proof/ (no wrfout/, scratch/, vram_*.csv) + scope.log
#   P/meta/             BUNDLE_USAGE.md (v0.4.3), session.log, rw3t_session.log, rw3t_topup_session.log
#   P/e41/icloud0/      the icloud=0 E41 scan (default scan is already P/e41)
# usage: bash assemble_receipts_v043.sh   (refuses if P/receipts or P/SHA256SUMS exists)
set -euo pipefail
P=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3
S=<DATA_ROOT>/wrf_gpu2/scratch/w43-release
L=<USER_HOME>/wrf_gpu2_lanes/w43-release
USAGE=<DATA_ROOT>/wrf_gpu2/worktrees/w43-release/proofs/w43_release/BUNDLE_USAGE.md
[ ! -e "$P/receipts" ] && [ ! -e "$P/SHA256SUMS" ] || { echo "refusing: $P/receipts or $P/SHA256SUMS exists"; exit 2; }
small=(check_after.json check_after.txt check_before.json check_before.txt device.err device.txt env.txt
       foreign_flags.txt mtimes.json rc.txt run.log source_head.txt t0_unix t_end_unix time.txt wrfout_sha.txt)
mkdir -p "$P/receipts" "$P/e41/icloud0"
for d in "$S"/gpu/runs/*/; do
  a=$(basename "$d"); mkdir -p "$P/receipts/$a/proof"
  for f in "${small[@]}"; do [ -e "$d/$f" ] && cp -p "$d/$f" "$P/receipts/$a/$f"; done
  cp -p "$d"/proof/* "$P/receipts/$a/proof/"
  cp -p "$S/gpu/runs/$a.scope.log" "$P/receipts/$a/scope.log"
done
cp -p "$USAGE" "$P/meta/BUNDLE_USAGE.md"
cp -p "$S/gpu/session.log" "$P/meta/session.log"
cp -p "$S/rw3t/session.log" "$P/meta/rw3t_session.log"
cp -p "$S/rw3t_topup/session.log" "$P/meta/rw3t_topup_session.log"
cp -p "$L"/followup/e41_icloud0/* "$P/e41/icloud0/"
# every phase-1 arm must carry its sha list, and it must equal the session copy byte for byte
for d in "$S"/gpu/runs/*/; do a=$(basename "$d"); cmp "$d/wrfout_sha.txt" "$P/receipts/$a/wrfout_sha.txt"; done
(cd "$P" && find bundle.env e41 meta receipts sealed tools -type f | LC_ALL=C sort | xargs -d '\n' sha256sum) > "$P/SHA256SUMS.tmp"
mv "$P/SHA256SUMS.tmp" "$P/SHA256SUMS"
echo "assembled: $(find "$P/receipts" -type f | wc -l) receipt files, SHA256SUMS $(wc -l < "$P/SHA256SUMS") lines sha $(sha256sum "$P/SHA256SUMS" | cut -c1-16)"
