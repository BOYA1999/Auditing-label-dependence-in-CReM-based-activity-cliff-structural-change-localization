import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--counterfactuals", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.representations / "manifest.json").read_text(encoding="utf-8"))
    unique = pd.read_csv(args.counterfactuals / "unique_scored_mutants_model.csv")
    failures = set(manifest["arms"]["unimol"]["conformer_failure_indices"])
    unique["unimol_2d_fallback"] = unique.index.isin(failures)
    unique[unique.unimol_2d_fallback].to_csv(args.output / "fallback_mutants.csv", index=False)
    scored = pd.read_csv(args.counterfactuals / "scored_mutants_model.csv")
    fallback_smiles = set(unique[unique.unimol_2d_fallback].canonical_smiles)
    scored["unimol_2d_fallback"] = scored.mutant_smiles.isin(fallback_smiles)
    masks = scored.groupby(["dataset", "row_id", "component_split", "mask_id"], as_index=False).agg(
        scored_mutants=("mutant_smiles", "size"), fallback_mutants=("unimol_2d_fallback", "sum")
    )
    masks["any_unimol_2d_fallback"] = masks.fallback_mutants > 0
    masks[masks.any_unimol_2d_fallback].to_csv(args.output / "affected_masks.csv", index=False)
    target = masks.groupby("dataset", as_index=False).agg(
        masks=("mask_id", "size"), affected_masks=("any_unimol_2d_fallback", "sum")
    )
    target["affected_mask_fraction"] = target.affected_masks / target.masks
    target.to_csv(args.output / "target_summary.csv", index=False)
    summary = {
        "status": "pass", "unique_mutants": len(unique), "fallback_unique_mutants": len(failures),
        "fallback_unique_mutant_fraction": len(failures) / len(unique),
        "scored_rows": len(scored), "fallback_scored_rows": int(scored.unimol_2d_fallback.sum()),
        "masks": len(masks), "affected_masks": int(masks.any_unimol_2d_fallback.sum()),
        "affected_mask_fraction": float(masks.any_unimol_2d_fallback.mean()),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
