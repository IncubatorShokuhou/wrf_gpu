# t-dfidiff verdict

**The original WRF DFL forecast runs with diff_opt/km_opt = 0/1 when the
DFI-specific options are omitted. The port already matches it.**

The original raw c49 input requests 1/4. DFL skips the backward initializer
which normally saves those options; forward initialization restores the
Registry-default DFI values instead. The print-only pristine solver probe and
all-frame original CPU header audit confirm the source trace. Details and
source locations are in [UPSTREAM_WRF_BUGS.md](UPSTREAM_WRF_BUGS.md).

Evidence:

- `receipts/TD01.json`: original CPU solver-entry probe; 6 observed rows across
  forward/forecast, every grid/solver diffusion pair 0/1; 1,810 byte-identical
  finite variable arrays across 5 paired files; 24 original c49 headers 0/1.
- `receipts/TD03.xml`: 15 targeted tests pass, including real host prepass and
  resume routing for omitted and explicit DFI diffusion. Numerical integration
  is stubbed in the new test; it proves orchestration, not physical fidelity.
- `receipts/TD06.log`: deleting restoration yields one genuine AssertionError,
  one explicit-1/4 control pass and zero setup errors.
- `RESULTS.json`: source/library/artifact hashes, clean-archive fixture check,
  exact-source reuse of the Swiss f672a573 and PROD d01 6fc715a2 release identities.

TD02's incomplete host fixture (missing QVAPOR), TD04's missing import path,
and TD05's late cpu_guard import are retained as failed setup attempts; they
are not mutation kills or model failures. TD03/TD06 correct those harness defects.

**Writer rule:** derive DIFF_OPT/KM_OPT from the **post-prepass forecast
namelist** used by the loader. Stage wrfinput headers are copied from the raw
input and can still contain 1/4. A direct raw-input `_load_domains` audit is
before DFI and cannot establish the forecast options. Do not change the model
or force an unconditional DFI-specific header override.

**v-prep / s2-integrate:** no dynamics changes and no c49 GPU rerun are required
by this result. c49's scheduled scientific GPU qualification remains required;
bind any existing GPU receipt to `dfi_input/namelist.input` and `dfi_report.json`
before reusing it. Header repair belongs to t-writer and must use those resolved
controls. This result does not claim c49 GPU/D6 qualification or header parity.

Performance/memory: all `src`, `data`, and `scripts` trees are byte-identical to
the already-measured release-identity source. No changed forecast kernels,
launches, carry fields, temporary arrays, transfers, cache keys, compile or
readiness work. Added files are tests and proof artifacts only. No GPU was used.
Disposable executables, paired forecast outputs, snapshot and test scratch
are removed after sealing their hashes and receipts.

Reproduce the independent CPU diagnostic on an empty lane-owned directory:

```sh
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 12 nice -n 19 \
  timeout 900 python proofs/t_dfidiff/oracle.py <USER_HOME>/wrf_gpu2_lanes/t-dfidiff/O2
```

The retained pristine build and read-only c49 reference/table paths in the
script must exist. This is an original-WRF configuration proof, not a timing
benchmark or a 24-hour field-fidelity comparison.
