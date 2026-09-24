import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


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


def run(*arguments):
    subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = args.config.resolve()
    cfg = json.loads(config.read_text(encoding="utf-8"))
    spec = cfg["sentinel_selection_sensitivity"]
    experiment = resolve(spec["experiment_root"])
    analysis = resolve(spec["analysis_output"])
    targets = spec["representative_targets"]
    inputs = {key: resolve(value) for key, value in spec["inputs"].items()}
    scripts = ROOT / "research_v2/scripts"

    run(scripts / "prepare_sentinel_selection_sensitivity.py", "--config", config)
    run(scripts / "seed_sentinel_counterfactual_cache.py", "--config", config)
    run(
        scripts / "generate_counterfactuals.py",
        "--config", config,
        "--prepared", experiment / "prepared",
        "--db", inputs["crem_database"],
        "--output", experiment / "counterfactuals",
        "--targets", *targets,
    )
    run(scripts / "assemble_sentinel_mutant_representations.py", "--config", config)
    run(
        scripts / "score_sanity_explanations.py",
        "--config", config,
        "--prepared", experiment / "prepared",
        "--representations", inputs["original_representations"],
        "--counterfactuals", experiment / "counterfactuals",
        "--mutant-representations", experiment / "mutant_representations",
        "--heads", inputs["model_heads"],
        "--output", experiment / "explanations",
        "--targets", *targets,
        "--seeds", "42",
    )
    run(scripts / "analyze_sentinel_selection_sensitivity.py", "--config", config)
    run(scripts / "validate_sentinel_selection_sensitivity.py", "--config", config)

    stage_paths = {
        "preparation": experiment / "prepared/feasibility.json",
        "counterfactuals": experiment / "counterfactuals/summary.json",
        "representations": experiment / "new_mutants/summary.json",
        "explanations": experiment / "explanations/summary.json",
        "analysis": analysis / "summary.json",
        "validation": analysis / "strict_validation.json",
    }
    stages = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in stage_paths.items()}
    accepted = {"pass", "success"}
    status = "pass" if all(stage.get("status") in accepted for stage in stages.values()) else "fail"
    manifest = {
        "run_id": cfg["run_id"],
        "status": status,
        "config_sha256": sha256(config),
        "stage_summary_sha256": {name: sha256(path) for name, path in stage_paths.items()},
    }
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
