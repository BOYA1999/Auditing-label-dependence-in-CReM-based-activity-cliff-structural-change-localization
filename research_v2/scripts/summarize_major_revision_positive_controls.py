import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gate_path = args.input / "calibrated_positive_control_gate.csv"
    rows_path = args.input / "calibrated_positive_controls.csv"
    gate = pd.read_csv(gate_path).sort_values(["control", "representation", "predictor", "explainer"])
    rows = pd.read_csv(rows_path)
    expected = {"halogen_count": 12, "aromatic_atom_count": 10, "heteroatom_count": 12}
    summaries = []
    for control, frame in gate.groupby("control", sort=True):
        values = frame.median_calibrated_nap.to_numpy(float)
        summaries.append({
            "control": control,
            "eligible_tasks": int(len(frame)),
            "expected_tasks": expected[control],
            "passing_tasks": int(frame.passes.astype(bool).sum()),
            "passing_fraction": float(frame.passes.astype(bool).mean()),
            "median_configuration_calibrated_nap": float(np.median(values)),
            "minimum_configuration_calibrated_nap": float(values.min()),
            "maximum_configuration_calibrated_nap": float(values.max()),
            "molecule_rows": int(len(rows[rows.control == control])),
        })
    concept = pd.DataFrame(summaries)
    gate.to_csv(args.output / "positive_control_configuration_summary.csv", index=False)
    concept.to_csv(args.output / "positive_control_concept_summary.csv", index=False)
    status = (
        len(gate) == sum(expected.values())
        and gate.groupby("control").size().to_dict() == expected
        and len(rows) > 0
        and np.isfinite(rows.randomization_calibrated_nap).all()
    )
    summary = {
        "status": "pass" if status else "fail",
        "interpretation": "heterogeneous_construct_sensitivity",
        "eligible_tasks": int(len(gate)),
        "passing_tasks": int(gate.passes.astype(bool).sum()),
        "minimum_configuration_median_calibrated_nap": float(gate.median_calibrated_nap.min()),
        "inputs": {"gate_sha256": digest(gate_path), "rows_sha256": digest(rows_path)},
        "outputs": {
            "configuration_sha256": digest(args.output / "positive_control_configuration_summary.csv"),
            "concept_sha256": digest(args.output / "positive_control_concept_summary.csv"),
        },
    }
    (args.output / "positive_control_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    if not status:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
