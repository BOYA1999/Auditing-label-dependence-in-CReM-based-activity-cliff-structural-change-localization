import argparse
import json
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--counterfactuals", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    model_cfg = cfg["representations"]["molformer"]
    snapshot = Path.home() / ".cache/huggingface/hub" / f"models--{model_cfg['model_id'].replace('/', '--')}" / "snapshots" / model_cfg["revision"]
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True)
    unique = pd.read_csv(args.counterfactuals / "unique_scored_mutants.csv")
    scored = pd.read_csv(args.counterfactuals / "scored_mutants.csv")
    coverage = pd.read_csv(args.counterfactuals / "mask_coverage.csv")
    lengths = [len(value) for value in tokenizer(unique.canonical_smiles.astype(str).tolist(), add_special_tokens=True, truncation=False)["input_ids"]]
    unique["molformer_tokens"] = lengths
    unique["within_limit"] = unique.molformer_tokens <= 202
    retained = unique[unique.within_limit].copy()
    excluded = unique[~unique.within_limit].copy()
    retained_smiles = set(retained.canonical_smiles)
    retained_scored = scored[scored.mutant_smiles.isin(retained_smiles)].copy()
    if retained_scored.mutant_smiles.nunique() != len(retained):
        raise ValueError("filtered scored/unique mutant mapping mismatch")
    counts = retained_scored.groupby(["row_id", "mask_id"]).size().rename("retained_scored_mutants")
    audit = coverage.merge(counts, on=["row_id", "mask_id"], how="left", validate="one_to_one")
    audit["retained_scored_mutants"] = audit.retained_scored_mutants.fillna(0).astype(int)
    audit["covered_after_length"] = audit.retained_scored_mutants > 0
    unique.to_csv(args.counterfactuals / "mutant_molformer_length_audit.csv", index=False)
    excluded.to_csv(args.counterfactuals / "mutant_molformer_length_exclusions.csv", index=False)
    retained.drop(columns=["molformer_tokens", "within_limit"]).to_csv(args.counterfactuals / "unique_scored_mutants_model.csv", index=False)
    retained_scored.to_csv(args.counterfactuals / "scored_mutants_model.csv", index=False)
    audit.to_csv(args.counterfactuals / "mask_coverage_after_length.csv", index=False)
    test = audit[audit.component_split == "test"]
    summary = {
        "status": "pass" if audit.covered_after_length.mean() >= 0.9 and test.covered_after_length.mean() >= 0.9 else "fail",
        "unique_mutants": int(len(unique)),
        "retained_unique_mutants": int(len(retained)),
        "excluded_unique_mutants": int(len(excluded)),
        "maximum_observed_tokens": int(max(lengths)),
        "scored_rows": int(len(scored)),
        "retained_scored_rows": int(len(retained_scored)),
        "mask_coverage_after_length": float(audit.covered_after_length.mean()),
        "test_mask_coverage_after_length": float(test.covered_after_length.mean()),
        "policy": "exclude overlength mutants from every representation and explainer; no truncation",
    }
    (args.counterfactuals / "molformer_length_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    if summary["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
