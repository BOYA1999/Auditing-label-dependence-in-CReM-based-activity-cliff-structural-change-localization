import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

import crem
from crem import frag_to_env_mp, fragmentation, import_env_to_db


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "research_v2/artifacts/experiment/reprxai_cliffs_pilot_v1/prepared/replacement_library.smi"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--max-core-heavy-atoms", type=int, default=12)
    parser.add_argument("--ncpu", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output / "source.smi"
    with args.input.open(encoding="utf-8") as src, source.open("w", encoding="utf-8", newline="\n") as dst:
        for idx, line in enumerate(src):
            if args.limit is not None and idx >= args.limit:
                break
            dst.write(line)

    frags = args.output / "fragmented.csv"
    env = args.output / "environments.csv"
    counted = args.output / "environments.counted.csv"
    db = args.output / "replacement_library.db"
    fragmentation.main(str(source), str(frags), 0, "\t", args.ncpu, ",", True)
    frag_to_env_mp.main(str(frags), str(env), None, args.radius, False, args.max_core_heavy_atoms, args.ncpu, False, ",", True)
    counts = Counter()
    with env.open(encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value:
                counts[value] += 1
    with counted.open("w", encoding="utf-8", newline="\n") as handle:
        for value, count in sorted(counts.items()):
            handle.write(f"{count} {value}\n")
    import_env_to_db.main(str(counted), str(db), args.radius, True, args.ncpu, True)
    with sqlite3.connect(db) as connection:
        rows = connection.execute(f"SELECT COUNT(*) FROM radius{args.radius}").fetchone()[0]
        environments = connection.execute(f"SELECT COUNT(DISTINCT env) FROM radius{args.radius}").fetchone()[0]
    manifest = {
        "crem_version": getattr(crem, "__version__", "0.2.14"),
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "source_molecules": sum(1 for _ in source.open(encoding="utf-8")),
        "limit": args.limit,
        "radius": args.radius,
        "max_core_heavy_atoms": args.max_core_heavy_atoms,
        "database_rows": int(rows),
        "distinct_environments": int(environments),
        "database_sha256": sha256(db),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()

