import hashlib
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
info = json.loads((ROOT/'package_manifest.json').read_text())
inputs = json.loads((ROOT/'EXTENSION_INPUTS_MANIFEST.json').read_text())
if all((ROOT/name).is_file() and hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == item['sha256'] for name,item in inputs.items()):
    print('All extension inputs match their SHA-256 hashes.')
else:
    with tempfile.TemporaryDirectory(prefix='crem_extension_') as folder:
        archive = Path(folder)/'inputs.zip'
        digest = hashlib.sha256()
        with urllib.request.urlopen(info['extension_input_url'],timeout=120) as response, archive.open('wb') as output:
            while chunk := response.read(1024*1024):
                output.write(chunk)
                digest.update(chunk)
        assert archive.stat().st_size == info['extension_input_archive_bytes']
        assert digest.hexdigest() == info['extension_input_archive_sha256']
        with zipfile.ZipFile(archive) as handle:
            assert set(handle.namelist()) == set(inputs)
            for name,item in inputs.items():
                assert (ROOT/name).resolve().is_relative_to(ROOT)
                handle.extract(name,ROOT)
                path = ROOT/name
                assert path.stat().st_size == item['bytes']
                assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'],name
    print('Restored extension inputs and verified every SHA-256 hash.')
