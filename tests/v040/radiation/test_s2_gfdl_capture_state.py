"""Native stream reader must retain nonzero input state and INTEGER counters."""
import json
from pathlib import Path
import numpy as np


def test_native_inout_stream_is_not_replaced_by_output(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root/'proofs/s2_flggfdl'))
    from capture_gfdl_lw import read
    from qualify_gfdl_sw import arguments
    meta=dict(inputs=['counter'],outputs=['counter'],co2_tables=[],
              arguments=['counter'],types={'counter':'i'},shapes={'counter':['1']})
    manifest=tmp_path/'stream.json';manifest.write_text(json.dumps(meta))
    raw=tmp_path/'stream.bin'
    with raw.open('wb') as f:
        np.asarray([3,7,8],np.int32).tofile(f)
    data,metadata=read(raw,manifest)
    assert data['counter'].dtype==data['out_counter'].dtype==np.int32
    assert data['counter'][0,0]==7 and data['out_counter'][0,0]==8
    assert arguments(data,0,metadata)[0].dtype==np.int32
    assert arguments(data,0,metadata)[0][0]==7
    del data['counter']
    output_buffer=arguments(data,0,metadata)[0]
    assert output_buffer.dtype==np.int32 and output_buffer[0]==0
