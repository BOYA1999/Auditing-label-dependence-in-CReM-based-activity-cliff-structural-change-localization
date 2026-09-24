import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--heads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    predictions = pd.read_csv(args.heads / "predictions.csv")
    models = pd.read_csv(args.heads / "model_manifest.csv")
    test = predictions[(predictions.variant == "label_permuted") & (predictions.component_split == "test")]
    keys = ["dataset", "representation", "predictor", "seed", "variant"]
    audit = test.groupby(keys, as_index=False).agg(
        test_rows=("prediction", "size"), unique_test_predictions=("prediction", "nunique"),
        test_prediction_min=("prediction", "min"), test_prediction_max=("prediction", "max"),
    )
    audit["test_prediction_range"] = audit.test_prediction_max - audit.test_prediction_min
    audit["endpoint_constant"] = audit.unique_test_predictions == 1
    audit["role"] = audit.dataset.map(
        lambda value: "development" if value in cfg["development_targets"] else "confirmation"
    )
    audit = audit.merge(models[keys + ["best_step", "path", "sha256"]], on=keys, validate="one_to_one")
    args.output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output / "randomized_endpoint_output_audit.csv", index=False)
    audit[audit.endpoint_constant].to_csv(args.output / "endpoint_constant_models.csv", index=False)
    summary = {
        "status": "revise_local_domain_gate",
        "models": int(len(models)), "expected_models": 1080,
        "randomized_models": int(len(audit)),
        "endpoint_constant_randomized_models": int(audit.endpoint_constant.sum()),
        "confirmation_endpoint_constant_models": int(((audit.endpoint_constant) & (audit.role == "confirmation")).sum()),
        "finite_predictions": bool(predictions.prediction.notna().all()),
        "retraining_allowed": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
