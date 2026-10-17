#!/usr/bin/env bash
# swiss_cpuwrf.sh [hours] [np] — original CPU-WRF (the reference's own binary + tables + runtime) on the box CPU.
set -uo pipefail
hours="${1:-3}"; np="${2:-4}"
ENVP="${WZ_CPUWRF_ENVP:-$HOME/micromamba/envs/wrf-build}"; C="${WZ_CPUWRF_DIR:-$HOME/src/wrf_gpu2_data/cpuwrf}"
REF="${WZ_SWISS_REF:-$HOME/src/wrf_gpu2_data/refs/swiss_rd11_cpu}"; TOOLS="${WZ_TOOLS:-$HOME/src/wrf_gpu2_env/tools}"
o="$WZ_RUN_DIR/cpuwrf_np${np}_$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$o"; cd "$o" || exit 3
echo "OUT=$o"
for f in "$C"/run/*; do
  case "$(basename "$f")" in *.exe|wrf|real|tc|ndown|namelist.input*|wrfinput_*|wrfbdy_*) continue;; esac
  ln -s "$f" .
done
rm -f wrf.exe; cp "$C/wrf.exe" ./wrf.exe   # never cp onto a symlink (it would overwrite the link target)
cp "$REF/wrfinput_d01" "$REF/wrfbdy_d01" .
python - "$REF/namelist.input" "$hours" <<'PY'
import datetime as dt, re, sys
src, hours = sys.argv[1], int(sys.argv[2])
t = open(src).read()
g = lambda k: int(re.search(rf"^\s*{k}\s*=\s*(\d+)", t, re.M).group(1))
s = dt.datetime(g("start_year"), g("start_month"), g("start_day"), g("start_hour"))
e = s + dt.timedelta(hours=hours)
for k, v in dict(run_days=0, run_hours=hours, end_year=e.year, end_month=e.month, end_day=e.day, end_hour=e.hour).items():
    t = re.sub(rf"^(\s*{k}\s*=\s*)\d+", rf"\g<1>{v}", t, count=1, flags=re.M)
open("namelist.input", "w").write(t)
PY
grep -E "^\s*(run_days|run_hours|end_day|end_hour)\s*=" namelist.input | tr -s ' ' | tr '\n' ' '; echo
export LD_LIBRARY_PATH="$ENVP/lib"
t0=$(date +%s.%N)
"$ENVP/bin/mpirun" --use-hwthread-cpus --bind-to none -np "$np" ./wrf.exe > mpirun.log 2>&1
rc=$?
t1=$(date +%s.%N)
echo "rc=$rc process_wall_s=$(python -c "print(round($t1-$t0,3))") success=$(grep -c 'SUCCESS COMPLETE WRF' rsl.out.0000 2>/dev/null)"
python "$TOOLS/probes/wrfout_timing.py" "$o" "$t0" --out "$o.timing.json" | grep TIMING_SUMMARY
[ "$rc" -ne 0 ] && { tail -20 mpirun.log; tail -20 rsl.error.0000 2>/dev/null; }
exit "$rc"
