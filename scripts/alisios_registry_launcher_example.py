"""20-line adoption template; metadata JSON holds real provenance and receipts."""
import json
import os
from pathlib import Path
import subprocess
import sys
from alisios_registry import Registry

start = json.loads(Path(sys.argv[1]).read_text())
end = json.loads(Path(sys.argv[2]).read_text())  # corpus/raw_deleted/extract SHA
smoke = os.environ.get('WRF_GPU2_SEQ_SMOKE') == '1'
registry = Registry(smoke=smoke, atlas_dir=os.environ.get('REGISTRY_PRIVATE_ATLAS') if smoke else None)
child_env = dict(os.environ, ALISIOS_WRF_RUN_ID=start['run_id'], ALISIOS_ATLAS_DIR=str(registry.atlas))
# The sequencer owns GPU lock/caps; smoke substitutes a bounded CPU command.
with registry.run(start, end=end) as outcome:  # terminal event in finally
    result = subprocess.run(sys.argv[3:], env=child_env, check=False)
    outcome.rc = result.returncode
# Set raw_deleted=True only after receipted cleanup; otherwise DONE_HELD.
# SIGKILL/power loss: backfill reviewed receipts; never invent a new start.
sys.exit(outcome.rc)
