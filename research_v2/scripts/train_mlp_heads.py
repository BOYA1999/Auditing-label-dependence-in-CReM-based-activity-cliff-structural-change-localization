import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import rdkit
import sklearn
from rdkit import Chem
from sklearn.metrics import mean_squared_error, r2_score
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "research_v2/configs/factorial_v1.json"
BASE = ROOT / "research_v2/artifacts/experiment/reprxai_cliffs_pilot_v1"
PREPARED = BASE / "prepared"
REPRESENTATIONS = BASE / "full/representations"
DEFAULT_OUT = BASE / "full/mlp_heads"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class MLP(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        layers = []
        current = input_dim
        for width in hidden:
            layers.extend([nn.Linear(current, width), nn.ReLU()])
            current = width
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, values):
        return self.network(values).squeeze(-1)


def load_representation(path, expected_ids):
    data = np.load(path, allow_pickle=False)
    row_ids = data["row_id"].astype(str)
    if not np.array_equal(row_ids, expected_ids):
        raise ValueError(f"row order mismatch: {path}")
    values = data["x"].astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite representation: {path}")
    return values


def atom_counts(smiles):
    atoms = list(Chem.MolFromSmiles(smiles).GetAtoms())
    return {
        "halogen_count": sum(atom.GetAtomicNum() in {9, 17, 35, 53} for atom in atoms),
        "heteroatom_count": sum(atom.GetAtomicNum() in {7, 8, 15, 16} for atom in atoms),
        "aromatic_atom_count": sum(atom.GetIsAromatic() for atom in atoms),
    }


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fit(values, target, train_mask, calibration_mask, params, seed, device):
    seed_everything(seed)
    feature_mean = values[train_mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_std = values[train_mask].std(axis=0, dtype=np.float64).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    target_mean = float(target[train_mask].mean())
    target_std = float(target[train_mask].std())
    if target_std < 1e-6:
        target_std = 1.0
    x_train = torch.from_numpy((values[train_mask] - feature_mean) / feature_std).to(device)
    y_train = torch.from_numpy(((target[train_mask] - target_mean) / target_std).astype(np.float32)).to(device)
    x_cal = torch.from_numpy((values[calibration_mask] - feature_mean) / feature_std).to(device)
    y_cal = torch.from_numpy(((target[calibration_mask] - target_mean) / target_std).astype(np.float32)).to(device)
    model = MLP(values.shape[1], params["hidden_dimensions"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    generator = torch.Generator(device=device).manual_seed(seed)
    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    stale = 0
    for epoch in range(params["maximum_epochs"]):
        model.train()
        order = torch.randperm(len(x_train), generator=generator, device=device)
        for start in range(0, len(order), params["batch_size"]):
            batch = order[start:start + params["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(x_train[batch]) - y_train[batch]) ** 2)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            calibration_loss = float(torch.mean((model(x_cal) - y_cal) ** 2).cpu())
        if calibration_loss < best_loss - params["minimum_improvement"]:
            best_loss = calibration_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= params["early_stopping_patience"]:
                break
    if best_state is None:
        raise RuntimeError("MLP did not produce a finite calibration state")
    model.load_state_dict(best_state)
    bundle = {
        "state_dict": best_state,
        "input_dimension": values.shape[1],
        "hidden_dimensions": params["hidden_dimensions"],
        "feature_mean": torch.from_numpy(feature_mean),
        "feature_std": torch.from_numpy(feature_std),
        "target_mean": target_mean,
        "target_std": target_std,
    }
    return model, bundle, best_epoch, best_loss


def predict(model, bundle, values, device, batch_size=2048):
    rows = []
    mean = bundle["feature_mean"].numpy()
    std = bundle["feature_std"].numpy()
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy((values[start:start + batch_size] - mean) / std).to(device)
            rows.append(model(batch).float().cpu().numpy())
    return np.concatenate(rows) * bundle["target_std"] + bundle["target_mean"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    model_dir = args.output / "models"
    model_dir.mkdir(exist_ok=True)
    molecules = pd.read_csv(PREPARED / "molecules.csv")
    pairs = pd.read_csv(PREPARED / "sentinel_pairs.csv")
    row_ids = molecules.row_id.astype(str).to_numpy()
    representations = {
        arm: load_representation(REPRESENTATIONS / f"{arm}.npz", row_ids)
        for arm in cfg["representations"]
    }
    targets = cfg["development_targets"][:1] if args.smoke else cfg["development_targets"]
    arms = cfg["representations"][:1] if args.smoke else cfg["representations"]
    seeds = [42] if args.smoke else cfg["seeds"]
    params = cfg["mlp_head"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = molecules.y.to_numpy(dtype=np.float32)
    prediction_rows = []
    metric_rows = []
    pair_rows = []
    model_rows = []
    for target in targets:
        target_mask = (molecules.dataset == target).to_numpy()
        train_mask = target_mask & molecules.component_split.isin(["train", "isolated"]).to_numpy()
        calibration_mask = target_mask & (molecules.component_split == "calibration").to_numpy()
        test_mask = target_mask & (molecules.component_split == "test").to_numpy()
        for arm in arms:
            values = representations[arm]
            for seed in seeds:
                model, bundle, best_epoch, best_loss = fit(values, y, train_mask, calibration_mask, params, seed, device)
                target_indices = np.flatnonzero(target_mask)
                predictions = predict(model, bundle, values[target_indices], device)
                prediction_map = {}
                for idx, value in zip(target_indices, predictions):
                    prediction_rows.append({
                        "row_id": row_ids[idx], "dataset": target,
                        "component_split": molecules.component_split.iloc[idx],
                        "representation": arm, "predictor": "mlp", "seed": seed,
                        "y": float(y[idx]), "prediction": float(value),
                    })
                    prediction_map[int(molecules.molecule_index.iloc[idx])] = float(value)
                for split, mask in (("calibration", calibration_mask), ("test", test_mask)):
                    observed = y[mask]
                    predicted = predict(model, bundle, values[mask], device)
                    metric_rows.append({
                        "dataset": target, "representation": arm, "predictor": "mlp", "seed": seed,
                        "split": split, "metric": "rmse",
                        "value": float(mean_squared_error(observed, predicted) ** 0.5), "n": int(mask.sum()),
                    })
                target_pairs = pairs[pairs.dataset == target]
                observed_lookup = molecules[molecules.dataset == target].set_index("molecule_index").y
                for split in ("calibration", "test"):
                    true_deltas = []
                    predicted_deltas = []
                    errors = []
                    for pair in target_pairs[target_pairs.component_split == split].itertuples(index=False):
                        true_delta = float(observed_lookup.loc[pair.left_index] - observed_lookup.loc[pair.right_index])
                        predicted_delta = prediction_map[int(pair.left_index)] - prediction_map[int(pair.right_index)]
                        error = abs(predicted_delta - true_delta)
                        true_deltas.append(true_delta)
                        predicted_deltas.append(predicted_delta)
                        errors.append(error)
                        pair_rows.append({
                            "dataset": target, "component_id": int(pair.component_id), "component_split": split,
                            "left_index": int(pair.left_index), "right_index": int(pair.right_index),
                            "representation": arm, "predictor": "mlp", "seed": seed,
                            "true_delta": true_delta, "predicted_delta": predicted_delta,
                            "absolute_signed_delta_error": error, "gap_score": -error,
                        })
                    metric_rows.extend([
                        {"dataset": target, "representation": arm, "predictor": "mlp", "seed": seed, "split": split, "metric": "signed_pair_delta_mae", "value": float(np.mean(errors)), "n": len(errors)},
                        {"dataset": target, "representation": arm, "predictor": "mlp", "seed": seed, "split": split, "metric": "pair_delta_pearson", "value": float(np.corrcoef(true_deltas, predicted_deltas)[0, 1]), "n": len(errors)},
                    ])
                model_path = model_dir / f"bioactivity--{target}--{arm}--seed{seed}.pt"
                torch.save(bundle, model_path)
                model_rows.append({
                    "kind": "bioactivity", "dataset": target, "representation": arm,
                    "predictor": "mlp", "seed": seed, "best_epoch": best_epoch,
                    "best_calibration_standardized_mse": best_loss,
                    "path": str(model_path.relative_to(ROOT)), "sha256": sha256(model_path),
                })
                print(f"bioactivity {target} {arm} seed={seed} best_epoch={best_epoch}", flush=True)
    controls = pd.DataFrame([atom_counts(value) for value in molecules.canonical_smiles])
    pooled_train = molecules.component_split.isin(["train", "isolated"]).to_numpy()
    pooled_calibration = (molecules.component_split == "calibration").to_numpy()
    pooled_test = (molecules.component_split == "test").to_numpy()
    control_rows = []
    for arm in arms:
        values = representations[arm]
        for control in controls.columns:
            target_values = controls[control].to_numpy(dtype=np.float32)
            model, bundle, best_epoch, best_loss = fit(values, target_values, pooled_train, pooled_calibration, params, 42, device)
            for split, mask in (("calibration", pooled_calibration), ("test", pooled_test)):
                predicted = predict(model, bundle, values[mask], device)
                control_rows.append({
                    "representation": arm, "predictor": "mlp", "control": control, "split": split,
                    "r2": float(r2_score(target_values[mask], predicted)),
                    "rmse": float(mean_squared_error(target_values[mask], predicted) ** 0.5), "n": int(mask.sum()),
                })
            model_path = model_dir / f"probe--{arm}--{control}.pt"
            torch.save(bundle, model_path)
            model_rows.append({
                "kind": "probe", "dataset": "pooled", "representation": arm,
                "predictor": "mlp", "control": control, "seed": 42,
                "best_epoch": best_epoch, "best_calibration_standardized_mse": best_loss,
                "path": str(model_path.relative_to(ROOT)), "sha256": sha256(model_path),
            })
            print(f"probe {arm} {control} best_epoch={best_epoch}", flush=True)
    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    pair_predictions = pd.DataFrame(pair_rows)
    probe_metrics = pd.DataFrame(control_rows)
    models = pd.DataFrame(model_rows)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    metrics.to_csv(args.output / "prediction_metrics.csv", index=False)
    pair_predictions.to_csv(args.output / "pair_predictions.csv", index=False)
    probe_metrics.to_csv(args.output / "probe_metrics.csv", index=False)
    models.to_csv(args.output / "model_manifest.csv", index=False)
    summary = {
        "run_id": cfg["run_id"], "smoke": args.smoke,
        "status": "success" if len(metrics) and np.isfinite(metrics.value).all() and np.isfinite(probe_metrics[["r2", "rmse"]]).all().all() else "failure",
        "bioactivity_models": int((models.kind == "bioactivity").sum()),
        "probe_models": int((models.kind == "probe").sum()),
        "test_rmse_by_representation": metrics[(metrics.split == "test") & (metrics.metric == "rmse")].groupby("representation").value.median().to_dict(),
        "probe_test_r2": {"|".join(key): float(value) for key, value in probe_metrics[probe_metrics.split == "test"].set_index(["representation", "control"]).r2.items()},
        "device": str(device),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "rdkit": rdkit.__version__, "scikit_learn": sklearn.__version__, "torch": torch.__version__},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    if summary["status"] != "success":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
