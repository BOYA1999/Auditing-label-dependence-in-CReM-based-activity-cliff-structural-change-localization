import argparse
import hashlib
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from train_common_heads import load_representation, model_from_config
from train_mlp_heads import fit as fit_mlp, predict as predict_mlp


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "research_v2/artifacts/experiment/xai_sanity_cliffs_full_v1"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def predict(predictor, model, bundle, values, device):
    if predictor == "xgboost":
        return model.predict(values)
    return predict_mlp(model, bundle, values, device)


def save_model(output, predictor, variant, target, representation, seed, model, bundle):
    suffix = "json" if predictor == "xgboost" else "pt"
    path = output / "models" / f"{variant}--{predictor}--{target}--{representation}--seed{seed}.{suffix}"
    if predictor == "xgboost":
        model.save_model(path)
    else:
        torch.save(bundle, path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "models").mkdir(exist_ok=True)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    molecules = pd.read_csv(args.prepared / "model_molecules.csv")
    pairs = pd.read_csv(args.prepared / "sentinel_pairs.csv")
    row_ids = molecules.row_id.astype(str).to_numpy()
    y = molecules.y.to_numpy(dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_targets = args.targets or cfg["targets"]
    if not set(selected_targets) <= set(cfg["targets"]):
        raise ValueError("requested target is absent from config")
    targets = selected_targets[:1] if args.smoke else selected_targets
    representations = list(cfg["representations"])[:1] if args.smoke else list(cfg["representations"])
    seeds = [42] if args.smoke else (args.seeds or cfg["seeds"])

    permuted_targets = {}
    permutation_rows = []
    for target in targets:
        target_index = cfg["targets"].index(target)
        target_mask = (molecules.dataset == target).to_numpy()
        eligible = target_mask & molecules.component_split.isin(["train", "isolated", "calibration"]).to_numpy()
        for replicate in seeds:
            permutation_seed = cfg["randomization"]["label_permutation_seed"] + 1000 * target_index + replicate
            fit_y = y.copy()
            fit_y[eligible] = np.random.default_rng(permutation_seed).permutation(y[eligible])
            permuted_targets[(target, replicate)] = fit_y
            for index in np.flatnonzero(eligible):
                permutation_rows.append({
                    "row_id": row_ids[index], "dataset": target, "replicate": replicate,
                    "permutation_seed": permutation_seed, "observed_y": float(y[index]),
                    "permuted_y": float(fit_y[index]),
                })
    permutation_frame = pd.DataFrame(permutation_rows)
    permutation_frame.to_csv(args.output / "permuted_labels.csv", index=False)

    prediction_rows = []
    metric_rows = []
    pair_rows = []
    model_rows = []
    for representation in representations:
        values = load_representation(args.representations / f"{representation}.npz", row_ids)
        for target in targets:
            target_mask = (molecules.dataset == target).to_numpy()
            train_mask = target_mask & molecules.component_split.isin(["train", "isolated"]).to_numpy()
            calibration_mask = target_mask & (molecules.component_split == "calibration").to_numpy()
            target_indices = np.flatnonzero(target_mask)
            observed = molecules[target_mask].set_index("molecule_index").y.to_dict()
            target_pairs = pairs[pairs.dataset == target]
            for predictor in ["xgboost", "mlp"]:
                for seed in seeds:
                    for variant, fit_y in [("trained", y), ("label_permuted", permuted_targets[(target, seed)])]:
                        if predictor == "xgboost":
                            model = model_from_config(cfg["common_head"], seed)
                            model.fit(
                                values[train_mask], fit_y[train_mask],
                                eval_set=[(values[calibration_mask], fit_y[calibration_mask])], verbose=False,
                            )
                            bundle = None
                            best_step = int(model.best_iteration)
                        else:
                            model, bundle, best_step, _ = fit_mlp(
                                values, fit_y, train_mask, calibration_mask,
                                cfg["mlp_head"], seed, device,
                            )
                        target_predictions = predict(predictor, model, bundle, values[target_indices], device)
                        if not np.isfinite(target_predictions).all():
                            raise ValueError(f"non-finite predictions: {target} {representation} {predictor} {seed} {variant}")
                        prediction_map = {}
                        for index, value in zip(target_indices, target_predictions):
                            prediction_map[int(molecules.molecule_index.iloc[index])] = float(value)
                            prediction_rows.append({
                                "row_id": row_ids[index], "dataset": target,
                                "component_split": molecules.component_split.iloc[index],
                                "representation": representation, "predictor": predictor,
                                "seed": seed, "variant": variant,
                                "observed_y": float(y[index]), "prediction": float(value),
                            })
                        for split in ["calibration", "test"]:
                            split_mask = target_mask & (molecules.component_split == split).to_numpy()
                            split_predictions = predict(predictor, model, bundle, values[split_mask], device)
                            metric_rows.append({
                                "dataset": target, "representation": representation,
                                "predictor": predictor, "seed": seed, "variant": variant,
                                "split": split,
                                "rmse_against_observed": float(mean_squared_error(y[split_mask], split_predictions) ** 0.5),
                                "prediction_std": float(np.std(split_predictions)),
                                "prediction_range": float(np.ptp(split_predictions)),
                                "n": int(split_mask.sum()),
                            })
                        for pair in target_pairs.itertuples(index=False):
                            true_delta = float(observed[pair.left_index] - observed[pair.right_index])
                            predicted_delta = prediction_map[int(pair.left_index)] - prediction_map[int(pair.right_index)]
                            pair_rows.append({
                                "dataset": target, "component_id": int(pair.component_id),
                                "component_split": pair.component_split,
                                "left_index": int(pair.left_index), "right_index": int(pair.right_index),
                                "representation": representation, "predictor": predictor,
                                "seed": seed, "variant": variant,
                                "true_delta": true_delta, "predicted_delta": predicted_delta,
                                "absolute_signed_delta_error": abs(predicted_delta - true_delta),
                            })
                        path = save_model(args.output, predictor, variant, target, representation, seed, model, bundle)
                        model_rows.append({
                            "dataset": target, "representation": representation,
                            "predictor": predictor, "seed": seed, "variant": variant,
                            "permutation_seed": (
                                cfg["randomization"]["label_permutation_seed"]
                                + 1000 * cfg["targets"].index(target) + seed
                                if variant == "label_permuted" else np.nan
                            ),
                            "best_step": best_step, "path": str(path.relative_to(ROOT)),
                            "sha256": sha256(path),
                        })
                        print(f"{representation} {target} {predictor} seed={seed} {variant} step={best_step}", flush=True)

    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    pair_predictions = pd.DataFrame(pair_rows)
    models = pd.DataFrame(model_rows)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    metrics.to_csv(args.output / "prediction_metrics.csv", index=False)
    pair_predictions.to_csv(args.output / "pair_predictions.csv", index=False)
    models.to_csv(args.output / "model_manifest.csv", index=False)
    randomized_test = metrics[(metrics.variant == "label_permuted") & (metrics.split == "test")]
    expected_models = len(targets) * len(representations) * 2 * len(seeds) * 2
    summary = {
        "run_id": cfg["run_id"], "smoke": args.smoke,
        "status": "pass" if len(models) == expected_models and (randomized_test.prediction_std > 1e-8).all() else "fail",
        "models": int(len(models)), "expected_models": expected_models,
        "finite_predictions": bool(np.isfinite(predictions.prediction).all()),
        "nonconstant_randomized_test_models": int((randomized_test.prediction_std > 1e-8).sum()),
        "randomized_test_models": int(len(randomized_test)),
        "minimum_randomized_test_std": float(randomized_test.prediction_std.min()),
        "trained_test_rmse_median": metrics[(metrics.variant == "trained") & (metrics.split == "test")].groupby(["representation", "predictor"]).rmse_against_observed.median().to_dict(),
        "input_hashes": {
            "config": sha256(args.config),
            "model_molecules": sha256(args.prepared / "model_molecules.csv"),
            "sentinel_pairs": sha256(args.prepared / "sentinel_pairs.csv"),
            "permuted_labels": sha256(args.output / "permuted_labels.csv"),
        },
    }
    summary["trained_test_rmse_median"] = {"|".join(key): float(value) for key, value in summary["trained_test_rmse_median"].items()}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    if summary["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
