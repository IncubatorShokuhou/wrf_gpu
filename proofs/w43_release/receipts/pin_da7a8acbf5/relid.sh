#!/usr/bin/env bash
# release identities Swiss + PROD on the v0.4.3 pin snapshot (CPU only), sequential in ONE RAM slot
cd <DATA_ROOT>/wrf_gpu2/scratch/w43-release/snap_da7a8acbf5
for arm in swiss prod; do
  if [ -e /tmp/wrf_gpu2_quiet ]; then while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 30; done; fi
  if [ $arm = swiss ]; then extra=""; else extra=<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725; fi
  /usr/bin/time -v env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONDONTWRITEBYTECODE=1 python3 proofs/s1_port/release_identity.py <DATA_ROOT>/wrf_gpu2/scratch/w43-release/snap_da7a8acbf5 <USER_HOME>/wrf_gpu2_lanes/w43-release/pin_da7a8acbf5/release_identity_$arm.json $extra > <USER_HOME>/wrf_gpu2_lanes/w43-release/pin_da7a8acbf5/relid_$arm.log 2>&1
  echo "$arm rc=$?" >> <USER_HOME>/wrf_gpu2_lanes/w43-release/pin_da7a8acbf5/relid_rc.txt
done
