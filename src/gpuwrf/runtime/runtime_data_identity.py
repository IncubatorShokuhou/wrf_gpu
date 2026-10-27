"""Host-only content binding for WRF tables, package data and runtime assets.

No mtime/content cache: a later identity must notice same-path replacement.
One identity deduplicates resolved files and hashes them in bounded chunks.
The reader inventory is checked in tests, independently of the AOT cheap key.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path


# WRF run-table families consumed by land/urban, radiation, cloud/aerosol and
# greenhouse-gas loaders. Include alternate-format/fallback tables so adding,
# deleting or changing a fallback cannot silently change the selected data.
# Lake constants currently live in Python; any future packaged lake table is
# covered automatically by the package-data walk and the reader coverage gate.
WRF_RUN_PATTERNS = ("*.TBL*", "*.tbl*", "*_DATA*", "*.dat*", "*.bin", "*.BIN",
                    "*.asc*", "*.formatted", "CAMtr_volume_mixing_ratio*", "tr[0-9]*", "co2_trans")
WRF_SOURCE_TABLES = ("phys/module_ra_rrtm.F", "phys/module_ra_rrtmg_lw.F", "phys/module_ra_rrtmg_sw.F")


def _readers(source_root):
    """Inventory filesystem operations, path builders and data-file literals.

    Exact AST signatures also detect an added read inside an existing loader.
    This scans every package module; Pallas memory loads and in-memory JSON
    decoding are not filesystem operations. Existing nonforecast/input/cache
    readers are inventoried too, so a new loader cannot hide under a new dir.
    """
    result = set()
    constructors = {"Path", "asset_path", "ensure_asset", "wrf_root", "wrf_run_dir",
                    "wrf_run_path", "wrf_path", "wrf_phys_path", "wrf_phys_dir"}
    methods = {"read_text", "read_bytes", "open", "joinpath", "with_name"}
    file_calls = {"numpy.load", "numpy.loadtxt", "numpy.fromfile", "numpy.genfromtxt",
                  "json.load", "pickle.load", "netCDF4.Dataset", "h5py.File"}
    root = Path(source_root)
    sources = [(path.relative_to(root).as_posix(), path) for path in sorted(root.rglob("*.py"))]
    manifest = root / "assets_manifest.json"
    if manifest.is_file():
        for name in json.loads(manifest.read_text())["assets"]:
            if name.endswith(".py"):
                candidates = (root.parents[1] / name, root / "_bundled" / name)
                path = next((p for p in candidates if p.is_file()), None)
                if path is None:
                    raise ValueError(f"runtime reader source is unavailable for coverage audit: {name}")
                sources.append(("asset/" + name, path))
    for relative, path in sources:
        tree = ast.parse(path.read_text())
        aliases = {}
        scopes = {}
        def locate(node, scope="<module>"):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                scope += "." + node.name
            scopes[id(node)] = scope
            for child in ast.iter_child_nodes(node):
                locate(child, scope)
        locate(tree)
        occurrences = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                aliases.update({a.asname or a.name: a.name for a in node.names})
            elif isinstance(node, ast.ImportFrom):
                aliases.update({a.asname or a.name: f"{node.module}.{a.name}" for a in node.names})
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                function = ast.unparse(node.func)
                first, _, rest = function.partition(".")
                qualified = aliases.get(first, first) + ("." + rest if rest else "")
                name = node.func.attr if isinstance(node.func, ast.Attribute) else function
                include = (name in methods or function in constructors or qualified in file_calls
                           or qualified.endswith(".Dataset") or function == "open")
                if include:
                    signature = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                    key = (scopes[id(node)], signature)
                    occurrences[key] = occurrences.get(key, 0) + 1
                    result.add(f"{relative}:{key[0]}:call:{signature}:{occurrences[key]}")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                if (value and not any(c.isspace() for c in value)
                        and ("." in value or value.endswith("_DATA") or ".TBL" in value)):
                    result.add(f"{relative}:{scopes[id(node)]}:path:{value}")
    return sorted(result)


def check_reader_coverage(source_root=None, registry_path=None):
    """Refuse a new/changed unreviewed file reader before it reaches runtime."""
    root = Path(source_root) if source_root is not None else Path(__file__).resolve().parents[1]
    registry = Path(registry_path) if registry_path is not None else Path(__file__).with_name("runtime_file_readers.json")
    expected = set(json.loads(registry.read_text())["reader_sites"])
    actual = set(_readers(root))
    new = sorted(actual - expected)
    if new:
        raise ValueError("runtime data reader lacks a reviewed content binding: " + "; ".join(new[:8]))
    return {"reader_sites": len(actual), "registered": len(expected), "unbound": []}


def runtime_data_identity(bundles):
    """Resolve the actual loader locations and bind their current bytes.

    Small package/WRF table families are conservative supersets of the selected
    schemes. Large remote microphysics assets use the loader's own scheme map;
    inactive hail/Jensen tables are not read just to fingerprint a Thompson run.
    Unmaterialized pinned remote assets are bound by their verified content pin,
    never fetched here; assets.ensure_asset refuses any other bytes on use.
    """
    from gpuwrf import assets
    from gpuwrf.config.paths import wrf_root
    package = Path(assets.__file__).resolve().parent
    files, seen = {}, {}

    def bind(key, path, *, pin=None, expected_bytes=None):
        path = Path(path)
        if not path.is_file():
            files[key] = ({"sha256": pin, "bytes": expected_bytes} if pin else
                          {"sha256": None, "storage": "absent"})
            return
        resolved = path.resolve(strict=True)
        if resolved not in seen:
            with resolved.open("rb") as stream:
                before = os.fstat(stream.fileno())
                value = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(block)
                after = os.fstat(stream.fileno())
            signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            if signature(before) != signature(after) or signature(after) != signature(path.stat()):
                raise ValueError(f"runtime table changed while binding content: {path}")
            seen[resolved] = {"sha256": value.hexdigest(), "bytes": after.st_size}
        files[key] = dict(seen[resolved])

    root = wrf_root()
    for pattern in WRF_RUN_PATTERNS:
        for path in sorted((root / "run").glob(pattern)):
            if path.is_file():
                bind("wrf/run/" + path.name, path)
    for relative in WRF_SOURCE_TABLES:
        bind("wrf/" + relative, root / relative)

    # This is the actual RRTMG source resolver, including GPUWRF_WRF_SRC,
    # central WRF root and the extractor's checkout/wheel fallback candidates.
    from gpuwrf.physics.rrtmg_lw import _extract_rrtmg_tables_module
    extractor = _extract_rrtmg_tables_module()
    for name in ("LW_SOURCE", "SW_SOURCE"):
        bind("rrtmg/" + name, getattr(extractor, name))
    for name in ("LW_DATA_CANDIDATES", "SW_DATA_CANDIDATES"):
        for index, path in enumerate(getattr(extractor, name)):
            bind(f"rrtmg/{name}/{index}", path)

    manifest = assets.manifest()
    codes = {int(bundle.namelist.mp_physics) for bundle in bundles.values()}
    selected = set(assets.tables_for_schemes(codes))
    for name, item in manifest["assets"].items():
        if item["kind"] == "bundle" or name in selected:
            bind("asset/" + name, assets.asset_path(name), pin=item["sha256"], expected_bytes=item["bytes"])
    directories = set()
    for directory, children, names in os.walk(package, followlinks=True):
        real = Path(directory).resolve(strict=True)
        if real in directories:
            children[:] = []  # directory aliases/cycles have already been covered
            continue
        directories.add(real)
        children[:] = sorted(name for name in children if name != "__pycache__")
        for name in sorted(names):
            path = Path(directory) / name
            if not path.is_file() or path.suffix in {".py", ".pyc", ".pyo"}:
                continue
            relative = path.relative_to(package).as_posix()
            asset_name = relative.removeprefix("_bundled/") if relative.startswith("_bundled/") else "src/gpuwrf/" + relative
            if asset_name not in manifest["assets"]:
                bind("package/" + relative, path)
    return {"schema_version": 1, "files": files}
