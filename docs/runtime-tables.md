# Runtime tables

Some physics schemes read large lookup tables (Thompson cold collection, hail mp=38, Jensen–Ishmael mp=55, …).
Tables larger than 10 MiB are not inside the Python wheel. They are published as assets of the matching GitHub
release and pinned by SHA-256 in `src/gpuwrf/assets_manifest.json`.

- **Where they come from:** neither the wheel nor the public git repository contains them. `gpuwrf run`
  downloads the tables of the selected microphysics scheme once before the model starts, from
  `https://github.com/wrf-gpu/wrf_gpu/releases/download/v<version>/<asset>`, checks the SHA-256 and stores them in
  `~/.cache/gpuwrf/tables/<sha256>/` (set `GPUWRF_TABLES_DIR` to use another directory, e.g. a shared or container
  volume). Later runs read the cached copy, which is checked against its SHA-256 once per run. Importing gpuwrf
  never downloads anything; library code that uses a table without `gpuwrf run` downloads it on first use.
- **Development checkouts** that still carry a table file use it in place.
- **Offline hosts:** on a connected machine run `python -m gpuwrf.assets fetch --all`, copy the cache directory to
  the offline host and point `GPUWRF_TABLES_DIR` at it. With `GPUWRF_TABLES_OFFLINE=1` gpuwrf never downloads.
  If a required table is missing, the run stops with a message naming the file, its SHA-256 and the download URL.
  It never runs with a scheme silently switched off.
- **Mirrors:** `GPUWRF_TABLES_URL` replaces the download URL template (`{version}`, `{asset}`, `{sha256}`).

`python -m gpuwrf.assets list` shows every table and where it is found; `python -m gpuwrf.assets verify` checks the
SHA-256 of the tables that are present. Where a table is stored does not change the compiled programs: the compile
cache is keyed by table content.
