#!/usr/bin/env bash
# o5-coast deletion-sensitivity mutants on a HARD-LINKED copy (sed -i rewrites by rename -> originals untouched).
set -u
W=<DATA_ROOT>/wrf_gpu2/worktrees/o5-coast; M=<DATA_ROOT>/wrf_gpu2/scratch/o5-coast/mut
run() { name=$1 file=$2 old=$3 new=$4
  [ -e "$M/$name" ] && { echo "exists $M/$name"; return 1; }; mkdir -p "$M/$name"
  (cd "$W" && for d in src tests scripts data pyproject.toml; do cp -al "$d" "$M/$name/"; done)
  python3 - "$M/$name/$file" "$old" "$new" <<'PY'
import sys
p,o,n=sys.argv[1:4]; s=open(p).read(); assert s.count(o)>=1, ('mutant anchor missing', o); s=s.replace(o,n)
import os; os.unlink(p); open(p,'w').write(s)
PY
  out=$(cd "$M/$name" && JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 timeout 900 taskset -c 8,9,12,24,25,28 nice -n 19 python -m pytest -q tests/test_o5_root_spec_walk.py --basetemp <USER_HOME>/wrf_gpu2_lanes/o5-coast/pytest_mut -p no:cacheprovider 2>&1 | grep -v "collected tests)")
  summary=$(echo "$out" | tail -1); nassert=$(echo "$out" | grep -c "AssertionError\|^E   *assert\|Error: Not equal\|Mismatched elements"); nerr=$(echo "$out" | grep -c " error")
  echo "$name | $summary | assertion_lines=$nassert setup_errors=$nerr"
}
run M7_native_mass_increments_zero src/gpuwrf/kernels/dyn_acoustic_fp32.py "        values=dict(mu=before.mu+cfg.dt*before.mu_spec_target,
                    mu_work=before.mu_work+cfg.dt*before.mu_spec_target,
                    muts=before.muts+cfg.dt*before.muts_spec_target,
                    theta_coupled_work=before.theta_coupled_work+cfg.dt*before.theta_spec_target)" "        values=dict(mu=before.mu,
                    mu_work=before.mu_work,
                    muts=before.muts,
                    theta_coupled_work=before.theta_coupled_work)"
run M8_native_uv_increments_zero src/gpuwrf/kernels/dyn_acoustic_fp32.py "            u=_pin_spec_ring_wrf_owned(u,before.u+cfg.dt*before.u_work_bdy,spec_zone=cfg.spec_zone)
            v=_pin_spec_ring_wrf_owned(v,before.v+cfg.dt*before.v_work_bdy,spec_zone=cfg.spec_zone)" "            u=_pin_spec_ring_wrf_owned(u,before.u,spec_zone=cfg.spec_zone)
            v=_pin_spec_ring_wrf_owned(v,before.v,spec_zone=cfg.spec_zone)"
run M9_native_ph_absolute src/gpuwrf/kernels/dyn_acoustic_fp32.py "    if before.ph_bdy_target is not None and before.ph_save_for_spec is not None:
        if cfg.nested_frozen_wrf_boundary_bundle or cfg.root_spec_walk:" "    if before.ph_bdy_target is not None and before.ph_save_for_spec is not None:
        if cfg.nested_frozen_wrf_boundary_bundle:"
rm -rf <DATA_ROOT>/wrf_gpu2/scratch/o5-coast/mut
