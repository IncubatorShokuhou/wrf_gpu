#!/usr/bin/env bash
# o5-coast GPU prep (CPU only): immutable git-archive export of COMMIT (src, scripts, data/fixtures, data/manifests,
# proofs/o5-coast) + data/wrf_pristine link, EXPORT_SHA256SUMS; WN3 0227 3-nest case dir of symlinks to our own CPU twin
# inputs (w43-twin0227, == teacher batch1 0227 input SHAs) with the ORIGINAL reference namelist (sha 8e77d97f);
# immutable launcher copy with SRC_SHA substituted.  usage: bash prepare.sh SRC_COMMIT [LAUNCHER_COMMIT]
set -euo pipefail
C=$(git -C <DATA_ROOT>/wrf_gpu2/worktrees/o5-coast rev-parse --verify "$1^{commit}")
L=<DATA_ROOT>/wrf_gpu2/scratch/o5-coast
EXP=$L/src_$C
T=<DATA_ROOT>/wrf_gpu2/scratch/w43-twin0227/C2_4x1
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
(( $(df --output=avail -B1G <DATA_ROOT> | tail -1) >= 367 )) || { echo "disk < 367 GiB (floor 365 + export)"; exit 4; }
if [ ! -e "$EXP" ]; then
  mkdir -p "$EXP"
  git -C <DATA_ROOT>/wrf_gpu2/worktrees/o5-coast archive "$C" src scripts data/fixtures data/manifests proofs/o5-coast | tar -x -C "$EXP"
  ln -s <USER_HOME>/src/wrf_pristine "$EXP/data/wrf_pristine"
  echo "$C" > "$EXP/EXPORT_COMMIT"
  (cd "$EXP" && find src scripts data/fixtures data/manifests proofs -type f -print0 | sort -z | xargs -0 sha256sum > EXPORT_SHA256SUMS)
  chmod -R a-w "$EXP/src" "$EXP/scripts" "$EXP/proofs"
fi
CASE=$L/case0227_3nest
mkdir -p "$CASE"
for f in wrfinput_d01 wrfinput_d02 wrfinput_d03 wrfbdy_d01 wrflowinp_d01 wrflowinp_d02 wrflowinp_d03; do ln -sfn "$T/cpu/$f" "$CASE/$f"; done
ln -sfn "$T/original_namelist.input" "$CASE/namelist.input"
# expected SHAs = teacher batch1 0227 (proofs/v043/w43-teacher/gpu/batch1_inputs_sha256.txt)
grep 'wn3_20260227/' <DATA_ROOT>/wrf_gpu2/worktrees/o5-coast/proofs/v043/w43-teacher/gpu/batch1_inputs_sha256.txt \
  | sed 's|wn3_20260227/||' > "$L/case0227_3nest.sha256"
(cd "$CASE" && sha256sum -c "$L/case0227_3nest.sha256")
# CUDA mini case: Swiss d01 example (our own RD11 inputs), SHAs from the F20 Swiss intake (inputs.json)
SW=$L/case_swiss; mkdir -p "$SW"
for f in namelist.input wrfbdy_d01 wrfinput_d01; do ln -sfn "<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu/$f" "$SW/$f"; done
python3 -c 'import json,sys; d=json.load(open(sys.argv[1]))["input_sha256"]; [print(v+"  "+k) for k,v in sorted(d.items())]' \
  <DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016/Swiss/inputs.json > "$L/case_swiss.sha256"
(cd "$SW" && sha256sum -c "$L/case_swiss.sha256")
LC=$(git -C <DATA_ROOT>/wrf_gpu2/worktrees/o5-coast rev-parse --verify "${2:-$C}^{commit}")
if [ ! -e "$L/tools_$LC" ]; then
  mkdir -p "$L/tools_$LC" && git -C <DATA_ROOT>/wrf_gpu2/worktrees/o5-coast archive "$LC" proofs/o5-coast | tar -x -C "$L/tools_$LC"
  chmod -R a-w "$L/tools_$LC/proofs"
fi
git -C <DATA_ROOT>/wrf_gpu2/worktrees/o5-coast show "$LC:proofs/o5-coast/gpu/coast_gpu.sh" \
  | sed -e "s/__SRC_SHA__/$C/" -e "s/__LAUNCH_SHA__/$LC/" > "$L/coast_gpu_$C.L$LC.sh"   # committed launcher, src = export
chmod a-w "$L/coast_gpu_$C.L$LC.sh"
sha256sum "$L/coast_gpu_$C.L$LC.sh"
du -sh "$EXP"
