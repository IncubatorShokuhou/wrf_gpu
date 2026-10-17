# Forecast for a region

`gpuwrf quickstart` turns a bounding box, grid spacing, UTC cycle and duration
into a single-domain WRF case. It uses original WPS 4.6.0 and WRF 4.7.1
`real.exe`, followed by the existing `gpuwrf run` command. Planning and
preprocessing use the CPU. Forecast execution needs an NVIDIA GPU and remains
subject to the normal CLI support, memory and scientific checks.

This onboarding path is experimental. CPU preprocessing qualification and
GPU forecast qualification are separate. New regions and the Mercator
projection are not covered by the released Alpine/Tenerife fidelity results.
Automatic nests and complete ECMWF forcing preparation are future work.

## Install the container

From this repository's root:

```bash
docker build -f containers/onboarding/Dockerfile -t wrfgpu-onboarding:local .
```

The recipe builds original WRF and WPS from pinned commits, with a single
CPU compiler job. It installs Python 3.11.14, JAX/jaxlib/CUDA plugin 0.10.0,
CUDA runtime 13.0.96, nvcc/ptxas 13.2.78 and Triton 3.6.0. The CUDA libraries
are constrained to the release workstation's captured distribution versions
in `containers/onboarding/constraints-cuda13.txt`; pinning JAX alone allows
newer CUDA components and is insufficient.
Geography is downloaded separately. The host supplies the NVIDIA driver and
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Builds require several GB of disk; the recipe retains source/runtime tables
and the pip package inventory at `/opt/gpuwrf/installed-requirements.txt`.
It does not publish an image. Debian package updates and transitive Python
dependencies can change; keep the built image digest for exact reuse.
The Python base image itself is pinned by digest.

To build just the CPU tools, use `--target preprocessing`. On this shared
development host, follow the kernel's QUIET check, CPU grant and GPU lock:

```bash
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
JAX_PLATFORMS=cpu DOCKER_BUILDKIT=0 taskset -c 28 nice -n 19 timeout 3600 \
  docker build --cpuset-cpus=28 --cpu-shares=64 --memory=4g \
  -f containers/onboarding/Dockerfile -t wrfgpu-onboarding:local .
```

The Dockerfile's compiler and installer commands run at nice 19; the container
CPU mask matters because affinity of the Docker client does not constrain
daemon-launched build containers. On other machines choose appropriate cores.

## Geography

For a small educational test:

```bash
python scripts/onboarding/download_geog.py "$PWD/geog"
mkdir -p work
```

This downloads UCAR's low-resolution mandatory package (about 150 MB
compressed), records its URL and SHA256, and deletes the downloaded archive
after extraction. Both transfer and expanded size are capped at 2 GB. The
destination must be new. The generated WPS namelist selects `lowres`.
The UCAR archive contains a top-level `WPS_GEOG_LOW_RES` directory. Mount
`geog/WPS_GEOG_LOW_RES`, containing `topo_gmted2010_5m`, `soiltype_top_5m`,
and the other datasets, as `/geog`.

These fields are for testing/education; at 1–3 km they do not resolve the
terrain or coastlines. For operational forecasts provide appropriate higher
resolution WPS geography and adjust `geog_data_res` in a manually prepared
case. The quickstart intentionally preserves its declared low-resolution
recipe. See [UCAR static data](https://www2.mmm.ucar.edu/wrf/users/download/get_sources_wps_geog.html).

## Plan first

For Tenerife, at 9 km:

```bash
docker run --rm -e JAX_PLATFORMS=cpu wrfgpu-onboarding:local quickstart \
  --bbox -17 28 -16 28.6 --resolution-km 9 \
  --date 2026-10-12T00 --hours 3 --vram-gib 12
```

Replace the date with a recent **available** GFS cycle. Dates are UTC,
formatted `YYYY-MM-DDTHH`; cycles are 00/06/12/18. The historical date above
identifies the CPU qualification experiment, not a permanently hosted case.
NOMADS is a rolling service; an expired or not-yet-published cycle fails with
the server error. There is no silent fallback to another initialization.

The four bbox coordinates are west, south, east, north in degrees. They
describe the area of interest. The generated domain surrounds it by at least
12 grid cells on each edge, with at least 40×40 mass cells. It uses WPS's
spherical Mercator projection (Earth radius 6,370,000 m), at the bbox's middle
latitude. `nx`/`ny` count mass cells; WRF `e_we`/`e_sn` are one larger. The
reference latitude is the center in Mercator coordinates. Supported spacings
are 1–30 km, bbox latitudes ±70°, and extents up to 40° longitude/30° latitude.
Dateline-crossing domains are refused. Smaller dt than the usual 6×dx(km)
may be chosen to divide every hour and the 10-minute radiation interval.

The memory estimate includes 44 mass levels, a 25% allowance and 1 GiB for
context/modules; 2 GiB of card capacity is kept outside it. Oversized requests
fail with advice. The planner never silently shrinks the bbox or increases
grid spacing. Terrain-driven CFL limits and actual compile memory still
need checks on the intended machine.

## Prepare and run

Prepare through `real.exe` and the CPU-only `gpuwrf run --dry-run`:

```bash
docker run --rm -e JAX_PLATFORMS=cpu \
  -v "$PWD/geog/WPS_GEOG_LOW_RES:/geog:ro" -v "$PWD/work:/work" \
  wrfgpu-onboarding:local quickstart \
  --bbox -17 28 -16 28.6 --resolution-km 9 \
  --date 2026-10-12T00 --hours 3 --vram-gib 12 \
  --case-dir /work/tenerife --prepare
```

The case directory must be new. Durations are multiples of three hours,
3–120 h. The launcher requests GFS 0.25° from
[NOAA NOMADS](https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl), retaining
all fields/levels in a geographic subset extending **beyond the padded domain**
by another 2°. It downloads 0,3,…,hours, including the final boundary endpoint.
All GRIB transfers together are capped at 2,000 MB; `--max-download-mb`
can lower that cap. Failed/partial downloads are not reused. Large requests
may exceed the cap and need a separately prepared WPS case.

Each time is checked for common pressure-level temperature, wind, height and
humidity, 2 m temperature/humidity, 10 m wind, surface pressure, mask, skin
temperature, terrain, snow, and all four GFS soil temperature/moisture layers.
Missing fields are reported by name. The check is metadata completeness;
original WPS and `real.exe` supply the actual meteorological interpolation.
No synthetic soil, copied deep layers or mixed initialization sources are
inserted. `num_metgrid_levels` is read from every generated `met_em` file;
it is not assumed to be a fixed GFS level count.

The generated physics is Thompson (`mp=8`), RRTMG (`ra=4/4`), MYNN
(`bl=5`, surface layer 5), Noah-MP (`sf=4`), and 2-D Smagorinsky
(`diff_opt=1`, `km_opt=4`). KF (`cu=1`) is selected at ≥9 km; finer grids
disable cumulus. The original tools create `wrfinput_d01`/`wrfbdy_d01`.
The launcher checks finite, unmasked floating fields, exact geometry/start,
area coverage and boundary interval records. WRF boundary records are
end-exclusive: the final metgrid time supplies the last interval's derivative.

Review `onboarding.json`, `grib/forcing.json`, and the `*.stage.log` files.
The manifest records input SHA256s, requested geometry, tool paths, forcing
URLs, field checks and the exact next forecast command. It reports
`PREPARED_CPU_ONLY`; that is not a completed forecast or a fidelity claim.

Run the prepared case on your GPU:

```bash
docker run --rm --gpus all \
  -v "$PWD/work:/work" wrfgpu-onboarding:local run \
  --input-dir /work/tenerife --output-dir /work/tenerife/forecast \
  --domain d01 --hours 3 --scratch-dir /work/tenerife/scratch
```

For one command, use `--run` instead of `--prepare`, remove the
`-e JAX_PLATFORMS=cpu` setting, and give the container `--gpus all`. The final
forecast calls the public CLI in the same process, retaining an inherited GPU
lock descriptor on hosts where it is required. Preserve the case path between
prepare and run. On the project development host, the entire GPU command must
use `scripts/with_gpu_lock.sh` with a manager lease; a container does not
inherit the lock proof automatically. The separate CPU preparation command is
the recommended path there. This wave permits **no GPU execution**.

Without Docker, install `pip install -e '.[onboard]'`, set `WPS_ROOT` to the
WPS source tree with its executable links, `GPUWRF_WRF_ROOT` to WRF 4.7.1,
and `WPS_GEOG` to the data directory. CLI flags override these paths.
For a CMake installation whose executables are elsewhere, use an owned WPS
directory with `geogrid.exe`, `ungrib.exe`, `metgrid.exe` links to the installed
binaries and `geogrid/`, `ungrib/`, `metgrid/` links to the read-only source
table directories. Nothing needs to write the shared WPS tree.

Other example areas (first plan, then prepare in new directories):

```bash
gpuwrf quickstart --bbox 7 46 8 47 --resolution-km 9 --date 2026-10-12T00 --hours 3
gpuwrf quickstart --bbox 150 -34 151 -33 --resolution-km 9 --date 2026-10-12T00 --hours 3
```

CPU qualification at the specified 2026-10-12 00Z cycle reached original
`real.exe` and the public CLI dry run for all three areas. Each generated
40×40×44 mass grids, 34 metgrid levels and both required forcing times (0/3 h).
Total GRIB transfers were 1.51/1.97/1.73 MB for Tenerife/Alps/Sydney; peak
process RSS was 247–250 MiB. All floating input fields were finite and the
requested area, start time and boundary records passed. These are CPU
preprocessing receipts, with GPU forecast qualification pending.

## ECMWF IFS and AIFS

Check actual open-data availability with a small live metadata download:

```bash
gpuwrf quickstart --bbox -17 28 -16 28.6 --date 2026-10-12T00 --hours 3 \
  --source aifs --check-source
```

This reads the ECMWF index, reports available and missing parameters, hashes
that index, and exits nonzero while preprocessing is unqualified. `--source
ifs` checks IFS. A local WPS-compatible GRIB2 file can instead be examined
with `--audit-grib FILE`. Its audit currently checks GFS-style four-layer
soil encodings; it is not a general ECMWF local-table decoder.

At the checked 2026-10-12 00Z cycle, IFS provides all four soil temperature
and moisture layers (`sot`/`vsw`, level type `sol`), but its current GRIB2
Vtable/caller path is not qualified. AIFS provides only layers 1/2 and lacks
layers 3/4 of both fields and `sd` (snow water equivalent). Recheck another
cycle instead of assuming this inventory remains unchanged. ALISIOS's read-only
`Vtable.AIFS_PURE` maps two soil layers; a Vtable cannot create missing data.
An actual IFS layer-4 GRIB probe uses soil code `2/3/18`, level type `151`
and CCSDS packing; the [bundled WPS 4.6 ECMWF Vtable](https://github.com/wrf-model/WPS/blob/v4.6.0/ungrib/Variable_Tables/Vtable.ECMWF)
expects a different deep-soil code and level encoding. Its existence does
not demonstrate compatibility with the current open-data GRIB2 files.
Automatic IFS/AIFS preparation therefore fails closed. Supplying complete,
coherent forcing and qualifying its Vtable/caller is required before enabling
that path. No credentialed ECMWF access is required for the inventory.

The [ECMWF access specification](https://confluence.ecmwf.int/spaces/DAC/pages/272310539/ECMWF+open+data+real-time+forecasts+from+IFS+and+AIFS)
documents current paths and byte-range retrieval. Its Cycle 50r1 naming
uses `oper` at all four cycle times; older 06/18 archives may use `scda`.
The [official Python client](https://github.com/ecmwf/ecmwf-opendata) can
retrieve selected fields for a separately prepared, complete source case.

## Consumer GPU planning profiles

These are **[I] inferred planning ceilings, not measured maximum grids**.
`gpuwrf quickstart --profiles` emits the precise values. The only baseline
behind this calculation is FINDINGS A58: PROD's 120×70 + 267×117 mass grids,
44 mass levels, 4.00 GB JAX live peak on the measured release predecessor.
It does not measure a single Mercator domain or these GPUs. Cold compilation,
radiation transients, desktop VRAM, host RAM and allocator fallback can limit
the actual runnable size. The normal CLI memory admission is retained.

| VRAM | Approximate square mass grid at 44 levels [I] | Status |
|---|---|---|
| 12 GiB | 276×276 | GPU measurement pending |
| 16 GiB | 332×332 | GPU measurement pending |
| 24 GiB | 422×422 | GPU measurement pending |
| 32 GiB | 496×496 | GPU measurement pending |

Do not interpret this table as a forecast speed or fidelity guarantee.
The [WRF WPS guide](https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/wps.html)
describes geography, Vtables, metgrid and boundary preprocessing in detail.
