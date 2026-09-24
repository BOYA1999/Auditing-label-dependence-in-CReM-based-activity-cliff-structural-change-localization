import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    for directory in args.directories:
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for arm in manifest["arms"]:
            path = directory / f"{arm}.npz"
            with np.load(path, allow_pickle=True) as data:
                row_id = np.asarray(data["row_id"].astype(str).tolist(), dtype="U")
                values = data["x"]
            np.savez_compressed(path, row_id=row_id, x=values)
            with np.load(path, allow_pickle=False) as check:
                if check["row_id"].dtype.kind != "U" or not np.isfinite(check["x"]).all():
                    raise ValueError(f"cache normalization failed: {path}")
            manifest["arms"][arm]["sha256"] = sha256(path)
            manifest["arms"][arm]["row_id_dtype"] = str(row_id.dtype)
        manifest["schema_repair"] = "row_id object array converted to fixed Unicode; representation values unchanged"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"directory": str(directory), "arms": list(manifest["arms"])}))


if __name__ == "__main__":
    main()
