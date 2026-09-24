import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures = set(manifest["arms"]["unimol"]["conformer_failure_row_ids"])
    evaluation = pd.read_csv(args.prepared / "evaluation_molecules.csv")
    pairs = pd.read_csv(args.prepared / "sentinel_pairs.csv")
    evaluation["unimol_2d_fallback"] = evaluation.row_id.isin(failures)
    pair_rows = []
    for pair in pairs.itertuples(index=False):
        left = f"{pair.dataset}:{int(pair.left_index)}"
        right = f"{pair.dataset}:{int(pair.right_index)}"
        pair_rows.append({
            "dataset": pair.dataset, "component_id": int(pair.component_id),
            "component_split": pair.component_split, "left_row_id": left, "right_row_id": right,
            "left_unimol_2d_fallback": left in failures,
            "right_unimol_2d_fallback": right in failures,
            "any_unimol_2d_fallback": left in failures or right in failures,
        })
    pair_frame = pd.DataFrame(pair_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(args.output / "evaluation_fallbacks.csv", index=False)
    pair_frame.to_csv(args.output / "pair_fallbacks.csv", index=False)
    summary = {
        "status": "pass",
        "model_rows": int(manifest["rows"]),
        "unimol_2d_fallback_rows": len(failures),
        "evaluation_rows": int(len(evaluation)),
        "evaluation_2d_fallback_rows": int(evaluation.unimol_2d_fallback.sum()),
        "sentinel_pairs": int(len(pair_frame)),
        "sentinel_pairs_with_2d_fallback": int(pair_frame.any_unimol_2d_fallback.sum()),
        "primary_policy": "retain documented Uni-Mol 2D fallback behavior",
        "sensitivity_policy": "exclude sentinel pairs touching a fallback endpoint",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
