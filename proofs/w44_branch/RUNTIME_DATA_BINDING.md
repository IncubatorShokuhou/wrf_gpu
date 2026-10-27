# CR11 runtime data binding

Restart/branch identity now includes `runtime_data`: content SHA256 and size of
the resolved runtime files. It is independent of the AOT cheap key. Files are
hashed freshly for every identity, in 1 MiB chunks; resolved aliases are hashed
once within that call. Same-path, same-size edits with restored mtime are caught.
No filesystem metadata cache substitutes for a content read.

The inventory covers conservative WRF/package data families and uses the actual
asset loader's scheme map for large remote microphysics tables. Missing remote
assets use the content pin enforced by `assets.ensure_asset`; identity remains
stable after materializing exactly those bytes. Identity construction downloads
nothing. Adding/removing fallback files also changes identity.

| Actual loader | Bound data |
| --- | --- |
| `io.land_history._landuse_init`, `io.lsm_static_extract` | LANDUSE.TBL, VEGPARM.TBL |
| `physics.noahmp.tables`, `io.noahmp_*` | MPTABLE.TBL, SOILPARM.TBL, GENPARM.TBL, option/soil/carbon/irrigation/hydrology tables |
| `physics.ruclsm` | VEGPARM.TBL, SOILPARM.TBL |
| `physics.urban_slucm`, `physics.urban_bep_static`, urban couplers | URBPARM.TBL/LCZ variants, MPTABLE.TBL |
| Lake initialization | Current constants are Python-source-bound; WRF TBL families and arbitrary package data also cover future lake resources |
| `physics.ra_lw_rrtm` | Actual WRF root module_ra_rrtm.F, RRTM_DATA[_DBL] |
| `physics.rrtmg_lw`, `physics.rrtmg_mp_re`, runtime extractor | Actual extractor LW/SW source paths and data candidates, including WRF_SRC/root/fallback selection; bundled RRTMG NPZ and extractor source |
| `physics.ra_cam_common`, `physics.ra_cam_sw`, `physics.wrf_clwrf_ghg` | CAM_ABS_DATA, CAM_AEROPT_DATA, CAMtr_volume_mixing_ratio and fallback variants |
| `physics.thompson_tables`, aerosol/hail modules | Manifest-managed Thompson base/cold/mixed/aerosol/collision/freezing tables selected by `assets.tables_for_schemes` |
| CAM ozone, Tegen aerosol | Common pinned data bundles |
| P3, NTU, Jensen, CAM-MG, NSSL, Ferrier, KF-CuP, QNSE, Goddard/FLG/GFDL | Package NPZ/JSON/data files; Jensen large asset selected by the actual scheme map |
| `physics.gfdleta_co2` | Package native initializer source and tr49t* data; generated grid/output scratch is derived, not a scientific input table |

Package resources include wheel `_bundled` data, extensionless files and
directory symlinks. Directory aliases/cycles are bounded. Python model source is
separately bound by the existing source SHA. WRF tables include `TBL`, `DATA`,
`dat`, binary, formatted/ASCII, greenhouse-gas and transmission-table families;
the actual RRTMG source/data resolver is also enumerated explicitly.

`src/gpuwrf/runtime/runtime_file_readers.json` is the reviewed coverage registry.
The scanner visits every package Python file and runtime Python asset, tracking
filesystem/path AST signatures, loader scope, repeated calls and data-path
literals. A new file reader, a new read in an existing loader, a cloned reader
expression in a different scope, or a repeated read cannot silently inherit an
old signature. New remote assets must also appear in the actual scheme map.
The coverage audit runs in tests, outside forecast execution. Registering a new
reader requires reviewing its content binding against the declared WRF/package/
asset, case-input, authenticated-restart or derived/offline category.

Evidence and resource limits are recorded in `RUNTIME_DATA_DELTA.json`. The
unmodified critic reproduction changes 555 LANDUSE rows and ALBBCK in all 1764
Swiss cells, while keeping namelists/output policy equal; the new identity
refuses the empty branch on `runtime_data`. Original case/pristine data remain
read-only. No forecast steps or GPU calls are used by that reproduction.

Measured host metadata cost on the mp8/common inventory: first call 1.274 s,
later median 0.169 s under shared CPU load; 148 logical entries / 313.56 MB
including aliases. Hashing uses at most two 1 MiB chunks concurrently. Process
peak RSS was 0.858 GiB including Python/JAX/table imports, not incremental hash
memory. This work runs once at checkpoint-enabled startup, adds no timestep
operations, new carry leaves or device buffers, and is skipped for an ordinary
run without checkpoint/restart/branch. GPU/RW gates remain pending.
