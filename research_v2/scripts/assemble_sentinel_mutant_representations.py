import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    spec = cfg["sentinel_selection_sensitivity"]
    inputs = {key: resolve(value) for key, value in spec["inputs"].items()}
    experiment = resolve(spec["experiment_root"])
    counterfactuals = experiment / "counterfactuals"
    source_scored = counterfactuals / "scored_mutants.csv"
    source_unique = counterfactuals / "unique_scored_mutants.csv"
    if not source_scored.is_file() or not source_unique.is_file():
        raise FileNotFoundError("counterfactual generation is incomplete")
    shutil.copy2(source_scored, counterfactuals / "scored_mutants_model.csv")
    shutil.copy2(source_unique, counterfactuals / "unique_scored_mutants_model.csv")

    alternate = pd.read_csv(source_unique)
    original = pd.read_csv(inputs["original_unique_mutants"])
    if alternate.row_id.duplicated().any() or alternate.canonical_smiles.duplicated().any():
        raise ValueError("alternate mutant index is not unique")
    if original.row_id.duplicated().any() or original.canonical_smiles.duplicated().any():
        raise ValueError("original mutant index is not unique")
    original_position = dict(zip(original.canonical_smiles.astype(str), range(len(original))))
    alternate["original_position"] = alternate.canonical_smiles.astype(str).map(original_position)
    alternate["reused"] = alternate.original_position.notna()
    missing = alternate[~alternate.reused][["canonical_smiles"]].copy().reset_index(drop=True)
    missing.insert(0, "row_id", [f"sentinel-new-mutant:{index}" for index in range(len(missing))])
    missing_position = dict(zip(missing.canonical_smiles.astype(str), range(len(missing))))
    alternate["new_position"] = alternate.canonical_smiles.astype(str).map(missing_position)

    extraction_root = experiment / "new_mutants"
    extraction_root.mkdir(parents=True, exist_ok=True)
    missing_path = extraction_root / "missing_unique_mutants.csv"
    missing.to_csv(missing_path, index=False)
    new_representations = extraction_root / "representations"
    if len(missing):
        manifest_path = new_representations / "manifest.json"
        reuse_extraction = False
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            reuse_extraction = manifest.get("input_sha256") == sha256(missing_path)
        if not reuse_extraction:
            command = [
                sys.executable,
                str(ROOT / "research_v2/scripts/extract_representations.py"),
                "--input", str(missing_path),
                "--output", str(new_representations),
                "--config", str(args.config),
                "--arms", *cfg["representations"].keys(),
            ]
            subprocess.run(command, cwd=ROOT, check=True)

    destination = experiment / "mutant_representations"
    destination.mkdir(parents=True, exist_ok=True)
    original_manifest = json.loads(
        (inputs["original_mutant_representations"] / "manifest.json").read_text(encoding="utf-8")
    )
    new_manifest = (
        json.loads((new_representations / "manifest.json").read_text(encoding="utf-8"))
        if len(missing) else {"arms": {}}
    )
    arm_manifest = {}
    reused_mask = alternate.reused.to_numpy(bool)
    for arm in cfg["representations"]:
        with np.load(inputs["original_mutant_representations"] / f"{arm}.npz", allow_pickle=False) as data:
            original_ids = data["row_id"].astype(str)
            if not np.array_equal(original_ids, original.row_id.astype(str).to_numpy()):
                raise ValueError(f"original {arm} representation order mismatch")
            original_values = data["x"].astype(np.float32, copy=False)
            dimension = original_values.shape[1]
            merged = np.empty((len(alternate), dimension), dtype=np.float32)
            merged[reused_mask] = original_values[alternate.loc[reused_mask, "original_position"].astype(int)]
            if len(missing):
                with np.load(new_representations / f"{arm}.npz", allow_pickle=False) as new_data:
                    new_ids = new_data["row_id"].astype(str)
                    if not np.array_equal(new_ids, missing.row_id.astype(str).to_numpy()):
                        raise ValueError(f"new {arm} representation order mismatch")
                    new_values = new_data["x"].astype(np.float32, copy=False)
                    merged[~reused_mask] = new_values[alternate.loc[~reused_mask, "new_position"].astype(int)]
        if not np.isfinite(merged).all():
            raise ValueError(f"non-finite merged {arm} representations")
        path = destination / f"{arm}.npz"
        np.savez_compressed(path, row_id=alternate.row_id.astype(str).to_numpy(dtype="U"), x=merged)
        arm_manifest[arm] = {
            "shape": list(merged.shape),
            "dtype": str(merged.dtype),
            "finite": True,
            "sha256": sha256(path),
            "reused_rows": int(reused_mask.sum()),
            "new_rows": int((~reused_mask).sum()),
        }
        del merged

    original_failure_indices = original_manifest["arms"]["unimol"].get("conformer_failure_indices", [])
    failed_smiles = set(original.iloc[original_failure_indices].canonical_smiles.astype(str))
    if len(missing):
        new_failure_indices = new_manifest["arms"]["unimol"].get("conformer_failure_indices", [])
        failed_smiles.update(missing.iloc[new_failure_indices].canonical_smiles.astype(str))
    alternate_failure_indices = [
        index for index, value in enumerate(alternate.canonical_smiles.astype(str)) if value in failed_smiles
    ]
    arm_manifest["unimol"]["conformer_failure_indices"] = alternate_failure_indices
    arm_manifest["unimol"]["conformer_failures"] = len(alternate_failure_indices)
    arm_manifest["unimol"]["conformer_failure_row_ids"] = [
        str(alternate.row_id.iloc[index]) for index in alternate_failure_indices
    ]
    manifest = {
        "input": str(source_unique.relative_to(ROOT)),
        "input_sha256": sha256(source_unique),
        "rows": len(alternate),
        "source_original_unique_mutants_sha256": sha256(inputs["original_unique_mutants"]),
        "missing_unique_mutants_sha256": sha256(missing_path),
        "arms": arm_manifest,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    audit = alternate[["row_id", "canonical_smiles", "reused"]].copy()
    audit["source"] = np.where(audit.reused, "frozen_original_cache", "new_extraction")
    audit.to_csv(extraction_root / "representation_reuse_audit.csv", index=False)
    scored = pd.read_csv(source_scored)
    status = "pass" if (
        scored.mutant_smiles.isin(set(alternate.canonical_smiles)).all()
        and len(audit) == len(alternate)
        and sum(item["new_rows"] for item in arm_manifest.values()) == len(missing) * len(arm_manifest)
    ) else "fail"
    summary = {
        "run_id": cfg["run_id"],
        "status": status,
        "alternate_unique_mutants": int(len(alternate)),
        "reused_unique_mutants": int(reused_mask.sum()),
        "new_unique_mutants": int((~reused_mask).sum()),
        "scored_mutant_rows": int(len(scored)),
        "unimol_conformer_fallback_rows": int(len(alternate_failure_indices)),
        "input_hashes": {
            "config": sha256(args.config),
            "scored_mutants": sha256(source_scored),
            "unique_mutants": sha256(source_unique),
            "original_unique_mutants": sha256(inputs["original_unique_mutants"]),
            "original_manifest": sha256(inputs["original_mutant_representations"] / "manifest.json"),
        },
        "output_hashes": {
            "reuse_audit": sha256(extraction_root / "representation_reuse_audit.csv"),
            "missing_unique_mutants": sha256(missing_path),
            "manifest": sha256(destination / "manifest.json"),
            **{arm: item["sha256"] for arm, item in arm_manifest.items()},
        },
    }
    (extraction_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
