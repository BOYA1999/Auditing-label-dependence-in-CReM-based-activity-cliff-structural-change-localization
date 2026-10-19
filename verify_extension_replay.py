import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
expected = json.loads((ROOT/'EXTENSION_REPLAY_EXPECTED.json').read_text())
errors = []
for name,item in expected.items():
    path = ROOT/name
    data = gzip.decompress(path.read_bytes()) if item['comparison']=='decompressed_csv' else path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=item['sha256']:
        errors.append(name)
print(json.dumps({'status':'pass' if not errors else 'failure','scientific_output_files':len(expected),'errors':errors},indent=2))
assert not errors
