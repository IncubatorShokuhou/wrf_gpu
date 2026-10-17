# E41 executable gate (CPU only)

Run after a GPU cold compile, before deleting its persistent cache:

```bash
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 13 nice -n 19 timeout 900 \
  python scripts/wave_cubin_gate.py CACHE OUT [JIT_GLOB] [--run-proof RUN_JSON]
```

Use your assigned CPU core. No GPU lease is needed: the scanner reads saved
executables and runs `cuobjdump` without creating a CUDA context. Put artifacts
on disk in your lane directory. `CUOBJDUMP` can select a disassembler; otherwise
it is resolved from `PATH`. Python needs the project's JAX/JAXlib and zstandard.

The frozen limits are STACK > 256 bytes, a non-leaf **callee** > 1000 SASS
instructions, or a kInput transpose fusion body > 500 optimized HLO instructions.
Parameters and the root count as HLO instructions. Large root callers and native
transpose SASS counts are diagnostics, not additional thresholds. Callee bounds
follow direct calls, branches, predicated returns and direct convergence-barrier
bindings local to each path. A BSYNC convergence target must be its following
instruction; unresolved targets or indirect flow are UNKNOWN.

Default discovery covers rich domain/static-scan/fused JIT entries plus every
domain/fused stepping AOT executable. JIT entries are zstd with a four-byte
compile-time prefix. AOT uses the IFRT/backend envelope, including Riegeli native
record 2. The matching executable supplies its own optimized HLO. Compacted
`slim.*` HLO remains UNKNOWN; native equality alone cannot establish HLO counts.
Auxiliary radiation-initialization AOT blobs are not stepping programs.

`OUT/wave_gate.json` reports `e41_gate` = PASS, FAIL or INCOMPLETE. Only PASS
exits zero. `hard_stack_gate` and `conservative_flags` remain for older readers;
a STACK pass alone is not a complete E41 pass. Missing declared programs, missing
ELFs, malformed cache/native data and empty disassembler inventories refuse the
scan and remove a previous verdict at that output path. Use a producer run proof
to detect deletion of an entire child program directory.

`scripts/gpu_smoke_e41.py CACHE OUT` delegates to the same full gate and also
writes the legacy `e41.json` filename. Per-cubin and HLO summaries remain on disk;
temporary cubins and the temporary SQLite compatibility file are removed.

Fixtures and controls live in `tests/v025/compile/test_e41_scanners.py` and
`test_wave_cubin_gate.py`. `fixtures/e41/tiny.cubin` is a committed native sample
assembled from the adjacent handwritten PTX with CUDA ptxas 13.1/sm_120; it needs
no GPU to inspect. Text fixtures cover spills, calls, branches and HLO counting.
