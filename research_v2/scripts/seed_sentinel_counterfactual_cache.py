import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path):
    if not path.is_file():
        return pd.DataFrame()
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return pd.DataFrame(rows)


def same_rows(left, right, keys):
    columns = sorted(set(left.columns) & set(right.columns))
    left = left[columns].sort_values(keys).reset_index(drop=True)
    right = right[columns].sort_values(keys).reset_index(drop=True)
    if "failure" in columns:
        left["failure"] = left.failure.fillna("")
        right["failure"] = right.failure.fillna("")
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, atol=1e-12, rtol=1e-12)
        return True
    except AssertionError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    spec = cfg["sentinel_selection_sensitivity"]
    experiment = resolve(spec["experiment_root"])
    destination = experiment / "counterfactuals"
    destination.mkdir(parents=True, exist_ok=True)
    completed_summary = destination / "summary.json"
    if completed_summary.is_file() and json.loads(completed_summary.read_text(encoding="utf-8")).get("status") == "success":
        print(json.dumps({"status": "pass", "action": "completed_counterfactuals_unchanged"}), flush=True)
        return

    original_root = ROOT / "research_v2/artifacts/experiment/xai_sanity_cliffs_full_v1/full/counterfactuals"
    evaluation = pd.read_csv(experiment / "prepared/evaluation_molecules.csv")
    audit_rows = []
    total_reused = 0
    total_current = 0
    for target in spec["representative_targets"]:
        desired = set(evaluation[evaluation.dataset == target].row_id.astype(str))
        source_dir = original_root / target
        target_dir = destination / target
        target_dir.mkdir(exist_ok=True)
        original_masks = pd.read_csv(source_dir / "mask_coverage.csv")
        original_mutants = pd.read_csv(source_dir / "mutants.csv")
        original_masks = original_masks[original_masks.row_id.astype(str).isin(desired)].copy()
        original_mutants = original_mutants[original_mutants.row_id.astype(str).isin(desired)].copy()

        current_masks = []
        current_mutants = []
        if (target_dir / "mask_coverage.csv").is_file():
            current_masks.append(pd.read_csv(target_dir / "mask_coverage.csv"))
        if (target_dir / "mutants.csv").is_file():
            current_mutants.append(pd.read_csv(target_dir / "mutants.csv"))
        current_masks.append(read_jsonl(target_dir / "partial/mask_rows.jsonl"))
        current_mutants.append(read_jsonl(target_dir / "partial/mutant_rows.jsonl"))
        current_masks = pd.concat([item for item in current_masks if len(item)], ignore_index=True) if any(len(item) for item in current_masks) else pd.DataFrame(columns=original_masks.columns)
        current_mutants = pd.concat([item for item in current_mutants if len(item)], ignore_index=True) if any(len(item) for item in current_mutants) else pd.DataFrame(columns=original_mutants.columns)
        if len(current_masks):
            current_masks = current_masks[current_masks.row_id.astype(str).isin(desired)].drop_duplicates(["row_id", "mask_id"])
        if len(current_mutants):
            current_mutants = current_mutants[current_mutants.row_id.astype(str).isin(desired)].drop_duplicates(["row_id", "mask_id", "mutant_smiles"])

        overlap = set(original_masks.row_id.astype(str)) & set(current_masks.row_id.astype(str))
        if overlap:
            left_masks = original_masks[original_masks.row_id.astype(str).isin(overlap)]
            right_masks = current_masks[current_masks.row_id.astype(str).isin(overlap)]
            left_mutants = original_mutants[original_mutants.row_id.astype(str).isin(overlap)]
            right_mutants = current_mutants[current_mutants.row_id.astype(str).isin(overlap)]
            if not same_rows(left_masks, right_masks, ["row_id", "mask_id"]):
                raise ValueError(f"regenerated masks differ from frozen cache for {target}")
            if not same_rows(left_mutants, right_mutants, ["row_id", "mask_id", "mutant_smiles"]):
                raise ValueError(f"regenerated mutants differ from frozen cache for {target}")

        masks = pd.concat([original_masks, current_masks], ignore_index=True).drop_duplicates(["row_id", "mask_id"])
        mutants = pd.concat([original_mutants, current_mutants], ignore_index=True).drop_duplicates(["row_id", "mask_id", "mutant_smiles"])
        masks = masks.sort_values(["row_id", "mask_id"]).reset_index(drop=True)
        mutants = mutants.sort_values(["row_id", "mask_id", "mutant_smiles"]).reset_index(drop=True)
        partial = target_dir / "partial"
        if partial.exists():
            backup = target_dir / "partial_before_cache_seed"
            if backup.exists():
                raise FileExistsError(f"cache-seed backup already exists: {backup}")
            shutil.move(str(partial), str(backup))
        masks.to_csv(target_dir / "mask_coverage.csv", index=False)
        mutants.to_csv(target_dir / "mutants.csv", index=False)
        reused_ids = set(original_masks.row_id.astype(str))
        current_ids = set(current_masks.row_id.astype(str)) - reused_ids
        total_reused += len(reused_ids)
        total_current += len(current_ids)
        audit_rows.append({
            "dataset": target,
            "desired_endpoints": len(desired),
            "reused_frozen_endpoints": len(reused_ids),
            "retained_newly_generated_endpoints": len(current_ids),
            "remaining_endpoints": len(desired - reused_ids - current_ids),
            "overlap_determinism_checked_endpoints": len(overlap),
        })
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(destination / "cache_seed_audit.csv", index=False)
    summary = {
        "status": "pass",
        "reused_frozen_endpoints": int(total_reused),
        "retained_newly_generated_endpoints": int(total_current),
        "remaining_endpoints": int(audit.remaining_endpoints.sum()),
        "overlap_determinism_checked_endpoints": int(audit.overlap_determinism_checked_endpoints.sum()),
    }
    (destination / "cache_seed_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
