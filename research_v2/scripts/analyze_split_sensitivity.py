import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def component_deltas(path, targets):
    frame = pd.read_csv(path)
    frame = frame[
        frame["dataset"].isin(targets)
        & frame["component_split"].eq("test")
        & frame["seed"].eq(42)
    ].copy()
    keys = ["dataset", "component_id", "representation", "predictor", "explainer"]
    if frame.duplicated(keys + ["variant"]).any():
        raise ValueError(f"duplicate model/explainer/component rows in {path}")
    wide = frame.pivot(index=keys, columns="variant", values="normalized_ap_lift").reset_index()
    if not {"trained", "label_permuted"}.issubset(wide.columns):
        raise ValueError(f"missing trained or label_permuted rows in {path}")
    wide["delta_nap"] = wide["trained"] - wide["label_permuted"]
    if not np.isfinite(wide[["trained", "label_permuted", "delta_nap"]].to_numpy()).all():
        raise ValueError(f"non-finite localization metric in {path}")
    observed_configs = wide.groupby(["dataset", "component_id"]).size()
    if not observed_configs.eq(12).all():
        raise ValueError(f"expected 12 configurations for every eligible component in {path}")
    return wide


def summarize(frame, draws, seed):
    target_estimates = (
        frame.groupby("dataset", as_index=False)
        .agg(median_delta_nap=("delta_nap", "median"), components=("component_id", "nunique"))
    )
    point = float(target_estimates["median_delta_nap"].median())
    by_target = {target: part for target, part in frame.groupby("dataset")}
    targets = np.asarray(sorted(by_target))
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled_targets = rng.choice(targets, size=len(targets), replace=True)
        estimates = []
        for target in sampled_targets:
            part = by_target[target]
            components = part["component_id"].drop_duplicates().to_numpy()
            sampled_components = rng.choice(components, size=len(components), replace=True)
            values = [part.loc[part["component_id"].eq(component), "delta_nap"].to_numpy() for component in sampled_components]
            estimates.append(float(np.median(np.concatenate(values))))
        bootstrap[draw] = float(np.median(estimates))
    ci = np.quantile(bootstrap, [0.025, 0.975])
    return target_estimates, point, [float(ci[0]), float(ci[1])], bootstrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "research_v2/configs/major_revision_split_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "research_v2/artifacts/analysis/xai_sanity_cliffs_major_revision_v1/split_controls")
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    targets = config["representative_targets"]
    sources = {
        "original_seed42": ROOT / "research_v2/artifacts/experiment/xai_sanity_cliffs_full_v1/full/sanity_explanations/pair_explanations.csv",
        **{
            f"split_seed_{split_seed}": ROOT / f"research_v2/artifacts/experiment/xai_sanity_cliffs_major_revision_v1/split_seed_{split_seed}/sanity_explanations/pair_explanations.csv"
            for split_seed in config["split_seeds"]
        },
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing split-analysis inputs: " + "; ".join(missing))
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    target_outputs = []
    output_paths = []
    for index, (scenario, path) in enumerate(sources.items()):
        frame = component_deltas(path, targets)
        target_estimates, point, ci, bootstrap = summarize(frame, args.draws, 20260903 + index)
        target_estimates.insert(0, "scenario", scenario)
        target_outputs.append(target_estimates)
        summaries.append({
            "scenario": scenario,
            "target_balanced_median_delta_nap": point,
            "ci95_low": ci[0],
            "ci95_high": ci[1],
            "targets": int(frame["dataset"].nunique()),
            "eligible_components": int(frame[["dataset", "component_id"]].drop_duplicates().shape[0]),
            "configuration_rows": int(len(frame)),
        })
        bootstrap_path = args.output / f"{scenario}_bootstrap.csv"
        pd.DataFrame({"draw": np.arange(args.draws), "target_balanced_median_delta_nap": bootstrap}).to_csv(bootstrap_path, index=False)
        output_paths.append(bootstrap_path)
    summary_path = args.output / "split_sensitivity_summary.csv"
    target_path = args.output / "split_target_estimates.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    pd.concat(target_outputs, ignore_index=True).to_csv(target_path, index=False)
    output_paths.extend([summary_path, target_path])
    manifest = {
        "status": "pass",
        "config": args.config.resolve().relative_to(ROOT.resolve()).as_posix(),
        "config_sha256": sha256(args.config),
        "bootstrap_draws": args.draws,
        "selection_rule": config["selection_rule"],
        "input_hashes": {scenario: sha256(path) for scenario, path in sources.items()},
        "output_hashes": {path.name: sha256(path) for path in output_paths},
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "scenarios": len(summaries), "targets": len(targets)}))


if __name__ == "__main__":
    main()
