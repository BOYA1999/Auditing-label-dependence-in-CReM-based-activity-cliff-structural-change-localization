import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
import sklearn
import xgboost
from rdkit import Chem
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "research_v2/configs/pilot_v1.json"
BASE = ROOT / "research_v2/artifacts/experiment/reprxai_cliffs_pilot_v1"
PREPARED = BASE / "prepared"
REPRESENTATIONS = BASE / "full/representations"
DEFAULT_OUT = BASE / "full/common_heads"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_representation(path, expected_ids):
    data = np.load(path, allow_pickle=False)
    row_ids = data["row_id"].astype(str)
    if not np.array_equal(row_ids, expected_ids):
        raise ValueError(f"row order mismatch: {path}")
    values = data["x"].astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite representation: {path}")
    return values


def model_from_config(params, seed):
    return XGBRegressor(
        random_state=seed,
        objective=params["objective"],
        n_estimators=params["n_estimators"],
        early_stopping_rounds=params["early_stopping_rounds"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_lambda=params["reg_lambda"],
        n_jobs=params["n_jobs"],
        tree_method=params["tree_method"],
        device=params["device"],
        eval_metric="rmse",
    )


def atom_counts(smiles):
    mol = Chem.MolFromSmiles(smiles)
    atoms = list(mol.GetAtoms())
    return {
        "halogen_count": sum(atom.GetAtomicNum() in {9, 17, 35, 53} for atom in atoms),
        "heteroatom_count": sum(atom.GetAtomicNum() in {7, 8, 15, 16} for atom in atoms),
        "aromatic_atom_count": sum(atom.GetIsAromatic() for atom in atoms),
    }


def fit_model(values, target, train_mask, calibration_mask, params, seed):
    model = model_from_config(params, seed)
    model.fit(
        values[train_mask],
        target[train_mask],
        eval_set=[(values[calibration_mask], target[calibration_mask])],
        verbose=False,
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    model_dir = args.output / "models"
    model_dir.mkdir(exist_ok=True)

    molecules = pd.read_csv(PREPARED / "molecules.csv")
    pairs = pd.read_csv(PREPARED / "sentinel_pairs.csv")
    row_ids = molecules.row_id.astype(str).to_numpy()
    row_lookup = {value: idx for idx, value in enumerate(row_ids)}
    representations = {
        arm: load_representation(REPRESENTATIONS / f"{arm}.npz", row_ids)
        for arm in cfg["representations"]
    }
    params = cfg["common_head"]
    prediction_rows = []
    metric_rows = []
    pair_rows = []
    model_rows = []

    for target in cfg["targets"]:
        target_mask = (molecules.dataset == target).to_numpy()
        train_mask = target_mask & molecules.component_split.isin(["train", "isolated"]).to_numpy()
        calibration_mask = target_mask & (molecules.component_split == "calibration").to_numpy()
        test_mask = target_mask & (molecules.component_split == "test").to_numpy()
        y = molecules.y.to_numpy(dtype=float)
        for arm, values in representations.items():
            for seed in cfg["seeds"]:
                model = fit_model(values, y, train_mask, calibration_mask, params, seed)
                target_indices = np.flatnonzero(target_mask)
                predictions = model.predict(values[target_indices])
                prediction_map = {}
                for idx, prediction in zip(target_indices, predictions):
                    prediction_rows.append(
                        {
                            "row_id": row_ids[idx],
                            "dataset": target,
                            "component_split": molecules.component_split.iloc[idx],
                            "representation": arm,
                            "seed": seed,
                            "y": y[idx],
                            "prediction": float(prediction),
                        }
                    )
                    prediction_map[int(molecules.molecule_index.iloc[idx])] = float(prediction)
                for split, mask in (("calibration", calibration_mask), ("test", test_mask)):
                    observed = y[mask]
                    predicted = model.predict(values[mask])
                    metric_rows.append(
                        {
                            "dataset": target,
                            "representation": arm,
                            "seed": seed,
                            "split": split,
                            "metric": "rmse",
                            "value": float(mean_squared_error(observed, predicted) ** 0.5),
                            "n": int(mask.sum()),
                        }
                    )
                target_pairs = pairs[pairs.dataset == target]
                for split in ("calibration", "test"):
                    errors = []
                    true_deltas = []
                    predicted_deltas = []
                    for pair in target_pairs[target_pairs.component_split == split].itertuples(index=False):
                        left_y = float(molecules.loc[(molecules.dataset == target) & (molecules.molecule_index == pair.left_index), "y"].iloc[0])
                        right_y = float(molecules.loc[(molecules.dataset == target) & (molecules.molecule_index == pair.right_index), "y"].iloc[0])
                        true_delta = left_y - right_y
                        predicted_delta = prediction_map[int(pair.left_index)] - prediction_map[int(pair.right_index)]
                        error = abs(predicted_delta - true_delta)
                        errors.append(error)
                        true_deltas.append(true_delta)
                        predicted_deltas.append(predicted_delta)
                        pair_rows.append(
                            {
                                "dataset": target,
                                "component_id": int(pair.component_id),
                                "component_split": split,
                                "left_index": int(pair.left_index),
                                "right_index": int(pair.right_index),
                                "representation": arm,
                                "seed": seed,
                                "true_delta": true_delta,
                                "predicted_delta": predicted_delta,
                                "absolute_signed_delta_error": error,
                                "gap_score": -error,
                            }
                        )
                    correlation = float(np.corrcoef(true_deltas, predicted_deltas)[0, 1])
                    metric_rows.extend(
                        [
                            {"dataset": target, "representation": arm, "seed": seed, "split": split, "metric": "signed_pair_delta_mae", "value": float(np.mean(errors)), "n": len(errors)},
                            {"dataset": target, "representation": arm, "seed": seed, "split": split, "metric": "pair_delta_pearson", "value": correlation, "n": len(errors)},
                        ]
                    )
                model_path = model_dir / f"bioactivity--{target}--{arm}--seed{seed}.json"
                model.save_model(model_path)
                model_rows.append(
                    {
                        "kind": "bioactivity",
                        "dataset": target,
                        "representation": arm,
                        "seed": seed,
                        "best_iteration": int(model.best_iteration),
                        "path": str(model_path.relative_to(ROOT)),
                        "sha256": sha256(model_path),
                    }
                )
                print(f"bioactivity {target} {arm} seed={seed} best={model.best_iteration}", flush=True)

    control_values = pd.DataFrame([atom_counts(value) for value in molecules.canonical_smiles])
    pooled_train = molecules.component_split.isin(["train", "isolated"]).to_numpy()
    pooled_calibration = (molecules.component_split == "calibration").to_numpy()
    pooled_test = (molecules.component_split == "test").to_numpy()
    control_rows = []
    for arm, values in representations.items():
        for control in control_values.columns:
            target_values = control_values[control].to_numpy(dtype=float)
            model = fit_model(values, target_values, pooled_train, pooled_calibration, params, 42)
            for split, mask in (("calibration", pooled_calibration), ("test", pooled_test)):
                predicted = model.predict(values[mask])
                control_rows.append(
                    {
                        "representation": arm,
                        "control": control,
                        "split": split,
                        "r2": float(r2_score(target_values[mask], predicted)),
                        "rmse": float(mean_squared_error(target_values[mask], predicted) ** 0.5),
                        "n": int(mask.sum()),
                    }
                )
            model_path = model_dir / f"probe--{arm}--{control}.json"
            model.save_model(model_path)
            model_rows.append(
                {
                    "kind": "probe",
                    "dataset": "pooled",
                    "representation": arm,
                    "control": control,
                    "seed": 42,
                    "best_iteration": int(model.best_iteration),
                    "path": str(model_path.relative_to(ROOT)),
                    "sha256": sha256(model_path),
                }
            )
            print(f"probe {arm} {control} best={model.best_iteration}", flush=True)

    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    pair_predictions = pd.DataFrame(pair_rows)
    controls = pd.DataFrame(control_rows)
    models = pd.DataFrame(model_rows)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    metrics.to_csv(args.output / "prediction_metrics.csv", index=False)
    pair_predictions.to_csv(args.output / "pair_predictions.csv", index=False)
    controls.to_csv(args.output / "probe_metrics.csv", index=False)
    models.to_csv(args.output / "model_manifest.csv", index=False)
    summary = {
        "run_id": cfg["run_id"],
        "status": "success" if np.isfinite(metrics.value).all() and np.isfinite(controls[["r2", "rmse"]]).all().all() else "failure",
        "bioactivity_models": int((models.kind == "bioactivity").sum()),
        "probe_models": int((models.kind == "probe").sum()),
        "test_rmse_by_representation": metrics[(metrics.split == "test") & (metrics.metric == "rmse")].groupby("representation").value.median().to_dict(),
        "test_pair_delta_mae_by_representation": metrics[(metrics.split == "test") & (metrics.metric == "signed_pair_delta_mae")].groupby("representation").value.median().to_dict(),
        "probe_test_r2": controls[controls.split == "test"].set_index(["representation", "control"]).r2.to_dict(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rdkit": rdkit.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    summary["probe_test_r2"] = {"|".join(key): float(value) for key, value in summary["probe_test_r2"].items()}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    if summary["status"] != "success":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
