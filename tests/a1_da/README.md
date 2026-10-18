Tests for `gpuwrf.assimilation.letkf` (lane a1-da). CPU only:
`JAX_PLATFORMS=cpu nice -n 19 taskset -c 8,9,24,25 python -m pytest tests/a1_da -q --basetemp <USER_HOME>/wrf_gpu2_lanes/<lane>/pytest`.
Real-file tests read ALISIOS/CPU-WRF paths read-only and skip when the files are absent.
