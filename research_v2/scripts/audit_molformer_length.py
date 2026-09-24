import argparse
import hashlib
import json
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
PREPARED = ROOT / "research_v2/artifacts/experiment/xai_sanity_cliffs_full_v1/prepared"
CONFIG = ROOT / "research_v2/configs/sanity_full_v1.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, default=PREPARED)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    model_cfg = cfg["representations"]["molformer"]
    snapshot = Path.home() / ".cache/huggingface/hub" / f"models--{model_cfg['model_id'].replace('/', '--')}" / "snapshots" / model_cfg["revision"]
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True)
    molecules = pd.read_csv(args.prepared / "molecules.csv")
    evaluation = pd.read_csv(args.prepared / "evaluation_molecules.csv")
    lengths = [len(value) for value in tokenizer(molecules.canonical_smiles.astype(str).tolist(), add_special_tokens=True, truncation=False)["input_ids"]]
    audit = molecules[["row_id", "dataset", "molecule_index", "component_split", "heavy_atoms"]].copy()
    audit["molformer_tokens"] = lengths
    audit["within_limit"] = audit.molformer_tokens <= int(202)
    excluded = audit[~audit.within_limit].copy()
    eligible_ids = set(audit.loc[audit.within_limit, "row_id"])
    model_molecules = molecules[molecules.row_id.isin(eligible_ids)].copy()
    audit.to_csv(args.prepared / "molformer_length_audit.csv", index=False)
    excluded.to_csv(args.prepared / "molformer_length_exclusions.csv", index=False)
    model_molecules.to_csv(args.prepared / "model_molecules.csv", index=False)
    summary = {
        "status": "pass" if not evaluation.row_id.isin(set(excluded.row_id)).any() else "fail",
        "max_tokens": 202,
        "molecules": int(len(molecules)),
        "eligible_shared_model_rows": int(len(model_molecules)),
        "excluded_shared_model_rows": int(len(excluded)),
        "excluded_evaluation_rows": int(evaluation.row_id.isin(set(excluded.row_id)).sum()),
        "maximum_observed_tokens": int(max(lengths)),
        "excluded_by_target_split": {"|".join(key): int(value) for key, value in excluded.groupby(["dataset", "component_split"]).size().items()},
        "policy": "exclude overlength rows from every representation and predictor; no truncation",
        "input_sha256": sha256(args.prepared / "molecules.csv"),
    }
    (args.prepared / "molformer_length_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    if summary["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
