"""Extract AMDGPU code objects (ELF64, e_machine 224) from a serialized XLA executable and report
per-kernel resources from their AMDHSA metadata notes (llvm-readelf --notes) plus instruction counts
(llvm-objdump -d). Lane o5-port, CPU only, read-only on the input blob.

usage: amdgpu_elfs.py BLOB OUT.json [--keep DIR]
"""
import json
import os
import re
import struct
import subprocess
import sys
import tempfile

LLVM = os.environ.get("O5_LLVM_BIN", "<DATA_ROOT>/wrf_gpu2/scratch/o5-port/venv_rocm/lib/python3.13/site-packages/"
                      "_rocm_sdk_core/lib/llvm/bin")


def elfs(blob):
    pos = 0
    while (off := blob.find(b"\x7fELF", pos)) >= 0:
        if blob[off + 4:off + 6] != b"\x02\x01" or struct.unpack_from("<H", blob, off + 18)[0] != 224:
            pos = off + 4
            continue
        shoff = struct.unpack_from("<Q", blob, off + 0x28)[0]
        she, shn = struct.unpack_from("<HH", blob, off + 0x3A)
        end = shoff + she * shn
        for k in range(shn):
            sec = off + shoff + k * she
            so, ss = struct.unpack_from("<QQ", blob, sec + 0x18)
            if struct.unpack_from("<I", blob, sec + 4)[0] != 8:
                end = max(end, so + ss)
        yield blob[off:off + end]
        pos = off + end


def kernel_rows(path):
    notes = subprocess.run([f"{LLVM}/llvm-readelf", "--notes", path], capture_output=True, text=True).stdout
    dis = subprocess.run([f"{LLVM}/llvm-objdump", "-d", path], capture_output=True, text=True).stdout
    rows = []
    # one entry per kernel under amdhsa.kernels ("  - .<key>:"); RDNA entries have no .agpr_count
    for block in [b for b in re.split(r"\n  - (?=\.)", notes)[1:] if ".symbol:" in b]:
        def num(key):
            m = re.search(r"\.%s:\s+(\d+)" % key, block)
            return int(m.group(1)) if m else None
        name = re.search(r"\.symbol:\s+(\S+?)(?:\.kd)?\s", block)
        rows.append({"kernel": name.group(1) if name else "?", "vgpr": num("vgpr_count"), "sgpr": num("sgpr_count"),
                     "vgpr_spill": num("vgpr_spill_count"), "sgpr_spill": num("sgpr_spill_count"),
                     "scratch": num("private_segment_fixed_size"), "lds": num("group_segment_fixed_size"),
                     "wavefront": num("wavefront_size")})
    ins = [ln.split()[0] for ln in dis.splitlines() if re.match(r"\s+[vs]_[a-z0-9_]+", ln)]
    stats = {"n_instr": len(ins), "v_fma_f32": sum(bool(re.match(r"v_(pk_)?fma[ckma]*_f32", i)) for i in ins),
             "v_mul_f32": sum(i.startswith("v_mul_f32") for i in ins), "f64": sum(i.endswith("_f64") for i in ins),
             "scratch_ops": sum(i.startswith("scratch_") for i in ins)}
    arch = re.search(r"amdhsa\.target:\s+\S*?(gfx[0-9a-z]+)", notes)
    return rows, stats, arch.group(1) if arch else None


def main():
    blob = open(sys.argv[1], "rb").read()
    keep = sys.argv[sys.argv.index("--keep") + 1] if "--keep" in sys.argv else None
    out = []
    for i, elf in enumerate(elfs(blob)):
        with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=False, dir=keep) as fh:
            fh.write(elf)
            path = fh.name
        try:
            rows, stats, arch = kernel_rows(path)
            out.append({"elf": i, "bytes": len(elf), "arch": arch, "kernels": rows, **stats})
        finally:
            if not keep:
                os.unlink(path)
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    print(json.dumps(out)[:1500])


if __name__ == "__main__":
    main()
