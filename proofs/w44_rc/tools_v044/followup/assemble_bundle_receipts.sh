#!/usr/bin/env bash
# w44-rc v0.4.4: copy the small receipts of the sealed v0.4.4 bundle into P (v0.4.3 layout) and write P/SHA256SUMS over
# bundle.env e41 meta receipts sealed tools (the cache is pinned by sealed/BUNDLE_MANIFEST.json). CPU, copies only; never
# touches P/jax, P/src or any wrfout. Run AFTER the critic verdict (manager), once.
#   P/receipts/<arm>/  every SHORT-entry arm: small files + proof/ (no wrfout/, scratch/, vram_*.csv) + scope.log
#   P/receipts/RW3/    the RW3 entry's per-arm receipts (receipts dir of the entry) + RW3T verdict files
#   P/e41/             the entry's E41 post (wave_gate.json per pass)
#   P/meta/            session logs, PREREG.json, PIN_MANIFEST.json, BUNDLE_USAGE.md (if given)
# usage: assemble_bundle_receipts.sh MANIFEST SHORT_RUN_DIR SHORT_RECEIPTS_DIR RW3_RECEIPTS_DIR [BUNDLE_USAGE.md]
set -euo pipefail
source <DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044/lib.sh
MAN=$1 SRUN=$2 SREC=$3 RREC=$4 USAGE=${5:-}
P=$("$W44R_PY" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["bundles"]["prod"]["dir"])' "$MAN")
[ -s "$P/sealed/BUNDLE_MANIFEST.json" ] || { echo "refusing: $P not sealed"; exit 2; }
[ ! -e "$P/SHA256SUMS" ] && [ -z "$(ls -A "$P/receipts" 2>/dev/null)" ] || { echo "refusing: $P/receipts or $P/SHA256SUMS exists"; exit 2; }
small=(check_after.json check_after.txt check_before.json check_before.txt device.err device.txt env_product.txt env_child.txt
       command.txt foreign_flags.txt mtimes.json rc.txt run.log source_head.txt t0_unix t_end_unix time.txt wrfout_sha.txt
       WRFOUT_FILES.tsv RECEIPT.json PROOF_SUMMARY.json timer.json)
mkdir -p "$P/receipts" "$P/e41"
for d in "$SRUN"/runs/*/; do
  a=$(basename "$d"); mkdir -p "$P/receipts/$a/proof"
  for f in "${small[@]}"; do [ -e "$d/$f" ] && cp -p "$d/$f" "$P/receipts/$a/$f"; done
  [ -d "$d/proof" ] && cp -p "$d"/proof/* "$P/receipts/$a/proof/" 2>/dev/null || true
  cp -p "$SRUN/runs/$a.scope.log" "$P/receipts/$a/scope.log"
  cmp "$d/wrfout_sha.txt" "$P/receipts/$a/wrfout_sha.txt"
done
cp -rp "$RREC" "$P/receipts/RW3"
cp -p "$SREC"/e41/* "$P/e41/"
cp -p "$SRUN/session.log" "$P/meta/session.log"
cp -p "$RREC/session.log" "$P/meta/rw3_session.log" 2>/dev/null || cp -p "$RREC/rw3t/session.log" "$P/meta/rw3_session.log"
cp -p "$(dirname "$MAN")/PREREG.json" "$MAN" "$P/meta/"
[ -n "$USAGE" ] && cp -p "$USAGE" "$P/meta/BUNDLE_USAGE.md"
(cd "$P" && find bundle.env e41 meta receipts sealed tools -type f | LC_ALL=C sort | xargs -d '\n' sha256sum) > "$P/SHA256SUMS.tmp"
mv "$P/SHA256SUMS.tmp" "$P/SHA256SUMS"
echo "assembled: $(find "$P/receipts" -type f | wc -l) receipt files, SHA256SUMS $(wc -l < "$P/SHA256SUMS") lines sha $(sha256sum "$P/SHA256SUMS" | cut -c1-16)"
