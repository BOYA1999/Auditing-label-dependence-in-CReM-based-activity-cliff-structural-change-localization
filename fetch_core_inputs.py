import hashlib
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
info = json.loads((ROOT / 'package_manifest.json').read_text())
inputs = json.loads((ROOT / 'CORE_INPUTS_MANIFEST.json').read_text())
if all((ROOT / name).is_file() and hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == item['sha256'] for name, item in inputs.items()):
    print('All four frozen inputs already match their SHA-256 hashes.')
else:
    with tempfile.TemporaryDirectory(prefix='cliff_inputs_') as folder:
        archive = Path(folder) / 'inputs.zip'
        digest = hashlib.sha256()
        with urllib.request.urlopen(info['core_input_url'], timeout=120) as response, archive.open('wb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
        assert archive.stat().st_size == info['core_input_archive_bytes'], 'Input archive size mismatch'
        assert digest.hexdigest() == info['core_input_archive_sha256'], 'Input archive SHA-256 mismatch'
        with zipfile.ZipFile(archive) as z:
            for name, item in inputs.items():
                assert (ROOT / name).resolve().is_relative_to(ROOT), 'Input path outside repository'
                z.extract(name, ROOT)
                restored = ROOT / name
                assert restored.stat().st_size == item['bytes'], name
                assert hashlib.sha256(restored.read_bytes()).hexdigest() == item['sha256'], name
    print('Restored four frozen inputs and verified every SHA-256 hash.')
