# Historical ALISIOS path audit

The delivered list is `<USER_HOME>/wrf_gpu2_lanes/w41-readaudit/HISTORICAL_ALISIOS_READS.json`.
Its first element contains scope, counts, exclusions and unknown-extent sources.
Subsequent elements identify source files, literal paths, date evidence, main
integration commits and receipt timestamps. This is a static inventory for the
manager's collective booking, not an authorization or a replay of those reads.

Repository inputs are frozen at `f9c48022f2629c38b21f64481e7a0df82313d886`.
Lane receipts are a live scan with individual content hashes. Receipt traversal
never follows symlinks; every parent and leaf open uses `O_NOFOLLOW`. No path
discovered in source or receipt content is opened, listed, statted or resolved.
Binary files, caches, snapshots and text over 8 MiB are excluded explicitly.

`UNKNOWN` means the evidence does not support a boolean conclusion. An issue
day does not bound the valid-day extent. Administrative dates in directory or
cross-message names are retained separately from forecast/valid days. A source
literal does not establish that its reader ran. Completed comparison receipts
are labeled separately, without inventing a read-spy receipt. Synthetic guard
paths are marked. Missing run dates stay UNKNOWN; filesystem mtime is not used.

Rebuild the history patch with a bounded, QUIET-guarded command on an allocated
lane CPU (the allocation can change; CPU 2 was granted for this audit):

```bash
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
JAX_PLATFORMS=cpu taskset -c 2 nice -n 19 timeout 180 git log \
  --first-parent --reverse --format='@@COMMIT %H %cI' -p --unified=0 \
  -G '/mnt/(data/alisios|server/work/src/alisios)/' \
  f9c48022f2629c38b21f64481e7a0df82313d886 -- tests scripts proofs \
  > <USER_HOME>/wrf_gpu2_lanes/w41-readaudit/main_path_history.patch
```

Build the list (stdlib only, no model import or compile cache):

```bash
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONDONTWRITEBYTECODE=1 \
  taskset -c 2 nice -n 19 timeout 240 \
  python3 proofs/w41_readaudit/build_inventory.py --repo . \
  --ref f9c48022f2629c38b21f64481e7a0df82313d886 \
  --history <USER_HOME>/wrf_gpu2_lanes/w41-readaudit/main_path_history.patch \
  --output <USER_HOME>/wrf_gpu2_lanes/w41-readaudit/HISTORICAL_ALISIOS_READS.json
```

`check_inventory.py` independently checks date boundaries, unknown extents,
adjacent literals, receipt timestamps and parent/leaf symlink refusal using
temporary harmless files under this lane's artifact directory. Removal of
nofollow handling and conversion of UNKNOWN to false both fail assertions.

The critic correction adds `context_dates.py`: a bounded, syntax-only pass for
constant Path joins/f-strings, concrete call/comprehension/decorator alternatives,
guard path/day mappings and same-object source/forecast-window pairs. It never
executes a reader, method, decorator or function. Unresolved expressions stay
placeholders. Constructed paths are labeled alternatives; they do not establish
executed reads. `/days/YYYY-MM-DD` is an explicit product day, while a dated case
directory still has an unknown additional valid extent. Execution `start_utc` and
`end_utc` are receipt timestamps and never forecast-window bounds.

The original handed-off list remains byte-for-byte at
`<USER_HOME>/wrf_gpu2_lanes/w41-readaudit/RA06_HISTORICAL_ALISIOS_READS.json`
(SHA256 `6fef26eaf0bf22bfcf6fc3d006f8dca1c3165d0841445bdff08c89bacc11eec5`).
The correction reuses that source/receipt inventory, checking all source hashes,
instead of repeating directory discovery. Reproduce it with the same guarded
Python command above, replacing `--history ...` with:

```text
--base-inventory <USER_HOME>/wrf_gpu2_lanes/w41-readaudit/RA06_HISTORICAL_ALISIOS_READS.json
```

`check_context_dates.py` checks independent binding/isolation fixtures and,
when passed the final list path, all 22 critic date cases and six plot execution
timestamp pairs against exact source hashes. Removing context extraction or
same-object window extraction causes genuine assertion failures. The summary
records the base hash and every added date or recovered run timestamp.
