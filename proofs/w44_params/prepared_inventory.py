"""Bind every prepared value/domain to its exact native files before E201."""
import hashlib
from pathlib import Path
import pickle


def artifact_bindings(status, cache):
    key = status["cheap_key"]
    hlo = status.get("hlo_sha256") or status.get("meta_hlo_sha256")
    assert key and hlo, "missing prepared key/HLO"
    parent = Path(status.get("aot_path") or status["blob_path"]).parent
    assert parent.resolve().is_relative_to(Path(cache).resolve())
    bindings = []
    digests = {}
    for stem in ("k_" + key, hlo):
        blob, meta_path = parent / (stem + ".xlaexec"), parent / (stem + ".meta")
        assert blob.is_file() and meta_path.is_file(), f"missing prepared artifact: {blob} / {meta_path}"
        with meta_path.open("rb") as stream:
            meta = pickle.load(stream)  # Own trusted AOT producer, CPU-only caller.
        assert meta.hlo_sha256 == hlo, "prepared HLO metadata mismatch"
        if stem.startswith("k_"):
            assert meta.cheap_key == key, "prepared cheap-key metadata mismatch"
        stat = blob.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity not in digests:
            with blob.open("rb") as stream:
                digests[identity] = hashlib.file_digest(stream, "sha256").hexdigest()
        digest = digests[identity]
        assert digest == meta.blob_sha256, "prepared blob integrity mismatch"
        bindings.append(dict(blob_path=str(blob), blob_sha256=digest,
                             meta_path=str(meta_path),
                             meta_sha256=hashlib.sha256(meta_path.read_bytes()).hexdigest(),
                             cheap_key=key, hlo_sha256=hlo))
    return bindings
