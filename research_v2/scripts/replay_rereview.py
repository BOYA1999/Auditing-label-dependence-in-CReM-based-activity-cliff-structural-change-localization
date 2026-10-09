import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'research_v2/artifacts/analysis/xai_sanity_cliffs_rereview_20261008'
expected = json.loads((DATA / 'manifest.json').read_text())
subprocess.run([sys.executable, '-B', str(ROOT / 'research_v2/scripts/analyze_mapping_exclusion.py')], cwd=ROOT, check=True)
checks = {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest for name, digest in expected.items()}
report = {'status': 'pass' if all(checks.values()) else 'failure', 'output_hashes': checks,
    'scope': 'Frozen-score six-case exclusion replay, including bootstrap and explicit common-atom/bond evidence. No model fitting.'}
(ROOT / 'rereview_replay_result.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
assert all(checks.values())
