import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def interval(values):
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def analyze(pairs, margin, draws, seed):
    keys = ["dataset", "component_id", "representation", "predictor", "explainer"]
    random_mean = pairs[pairs.variant == "label_permuted"].groupby(keys, as_index=False).normalized_ap_lift.mean().rename(
        columns={"normalized_ap_lift": "randomized_mean_nap"}
    )
    trained = pairs[pairs.variant == "trained"].merge(random_mean, on=keys, validate="many_to_one")
    trained["delta_sanity"] = trained.normalized_ap_lift - trained.randomized_mean_nap
    targets = {target: [group.delta_sanity.to_numpy() for _, group in frame.groupby("component_id")]
               for target, frame in trained.groupby("dataset")}
    names = sorted(targets)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(draws)
    for draw in range(draws):
        medians = []
        for target_index in rng.integers(0, len(names), len(names)):
            groups = targets[names[target_index]]
            sampled = [groups[index] for index in rng.integers(0, len(groups), len(groups))]
            medians.append(np.median(np.concatenate(sampled)))
        bootstrap[draw] = np.median(medians)
    target_estimates = trained.groupby("dataset").delta_sanity.median()
    component = pairs.groupby(keys + ["variant"], as_index=False).normalized_ap_lift.mean().pivot(
        index=keys, columns="variant", values="normalized_ap_lift"
    ).reset_index()
    component["delta_sanity"] = component.trained - component.label_permuted
    rng = np.random.default_rng(seed)
    equivalent = 0
    cells = 0
    for _, frame in component.groupby(["dataset", "representation", "predictor", "explainer"]):
        values = frame.delta_sanity.to_numpy()
        sampled = values[rng.integers(0, len(values), size=(draws, len(values)))]
        low, high = interval(np.median(sampled, axis=1))
        equivalent += low >= -margin and high <= margin
        cells += 1
    low, high = interval(bootstrap)
    return {
        "targets": len(names), "pair_rows": len(pairs), "trained_delta_rows": len(trained),
        "primary_median": float(np.median(target_estimates)), "ci_low": low, "ci_high": high,
        "primary_equivalent": low >= -margin and high <= margin,
        "equivalent_cells": int(equivalent), "cells": cells,
        "equivalent_cell_fraction": float(equivalent / cells),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--endpoint-constant", type=Path, required=True)
    parser.add_argument("--pair-fallbacks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    pairs = pd.read_csv(args.scores / "pair_explanations.csv")
    pairs = pairs[(pairs.dataset.isin(cfg["confirmation_targets"])) & (pairs.component_split == "test")].copy()
    constant = pd.read_csv(args.endpoint_constant)
    constant = constant[constant.role == "confirmation"][["dataset", "representation", "predictor"]].drop_duplicates()
    flagged_cells = set(map(tuple, constant.itertuples(index=False, name=None)))
    endpoint_keep = np.array([
        (row.dataset, row.representation, row.predictor) not in flagged_cells for row in pairs.itertuples()
    ])
    fallback = pd.read_csv(args.pair_fallbacks)
    fallback = fallback[(fallback.component_split == "test") & fallback.any_unimol_2d_fallback]
    flagged_pairs = set(map(tuple, fallback[["dataset", "component_id"]].itertuples(index=False, name=None)))
    fallback_keep = np.array([
        row.representation != "unimol" or (row.dataset, row.component_id) not in flagged_pairs
        for row in pairs.itertuples()
    ])
    scenarios = {
        "primary": np.ones(len(pairs), dtype=bool),
        "exclude_endpoint_constant_model_cells": endpoint_keep,
        "exclude_unimol_fallback_pairs": fallback_keep,
        "combined_exclusions": endpoint_keep & fallback_keep,
    }
    rows = []
    for name, keep in scenarios.items():
        rows.append({"scenario": name, **analyze(
            pairs[keep], float(cfg["inference"]["equivalence_margin"]),
            int(cfg["inference"]["bootstrap_draws"]), int(cfg["inference"]["bootstrap_seed"]),
        )})
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "sensitivity_summary.csv", index=False)
    summary = {
        "status": "success", "endpoint_constant_confirmation_model_cells": len(flagged_cells),
        "unimol_fallback_test_components": len(flagged_pairs), "scenarios": rows,
    }
    (args.output / "sensitivity_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
