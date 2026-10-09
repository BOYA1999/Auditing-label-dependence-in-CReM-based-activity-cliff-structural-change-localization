import csv
import hashlib
import json
import re
from pathlib import Path


root = Path(__file__).parent
manifest_path = root / "MANIFEST.sha256"
errors = []
entries = []
for line in manifest_path.read_text(encoding="utf-8").splitlines():
    try:
        expected, relative = line.split("  ", 1)
    except ValueError:
        errors.append(f"malformed manifest line: {line}")
        continue
    entries.append((relative, expected))

relatives = [relative for relative, _ in entries]
if relatives != sorted(relatives):
    errors.append("manifest paths are not sorted")
if len(relatives) != len(set(relatives)):
    errors.append("manifest contains duplicate paths")
if "MANIFEST.sha256" in relatives:
    errors.append("manifest must not list itself")

actual_files = sorted(
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file() and path != manifest_path and path.relative_to(root).parts[0] != ".git"
)
if actual_files != relatives:
    errors.append("manifest file set differs from package payload")

payload_bytes = 0
for relative, expected in entries:
    path = root / relative
    if not path.is_file():
        errors.append(f"missing file: {relative}")
        continue
    payload_bytes += path.stat().st_size
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        errors.append(f"hash mismatch: {relative}")

metadata = json.loads((root / "package_manifest.json").read_text(encoding="utf-8"))
if metadata["file_count_excluding_hash_manifest"] != len(entries):
    errors.append("package_manifest file count mismatch")
if metadata["total_bytes_excluding_hash_manifest"] != payload_bytes:
    errors.append("package_manifest byte count mismatch")

expected_figures = {
    "figure1a_chronology.csv", "figure1b_prediction_damage.csv",
    "figure1c_pairwise_signal.csv", "figure1d_scope.csv",
    "figure2a_model_absolute.csv", "figure2b_generator_null_adjusted.csv",
    "figure2c_support.csv", "figure2d_positive_controls.csv",
    "figure3a_estimands.csv", "figure3b_target_configuration_cells.csv",
    "figure3c_rationale_targets.csv", "figure3d_mapping_ambiguity.csv",
    "figure4a_target_configuration_heatmap.csv", "figure4b_hierarchical_intervals.csv",
    "figure4c_robustness_checks.csv", "figure4d_lime_failure.csv",
}
observed_figures = {path.name for path in (root / "figure_source_data").glob("*.csv")}
if observed_figures != expected_figures:
    errors.append("figure-source allowlist mismatch")

supplement_root = root / "research_v2/artifacts/analysis/xai_sanity_cliffs_supplement_tables_v1"
expected_supplements = {f"table_s{index:02d}.csv" for index in range(1, 27)}
observed_supplements = {path.name for path in supplement_root.glob("*.csv")}
if observed_supplements != expected_supplements:
    errors.append("supplementary-table allowlist mismatch")
supplement_manifest = json.loads((supplement_root / "manifest.json").read_text(encoding="utf-8"))
supplement_checks = {}
for item in supplement_manifest["tables"]:
    path = supplement_root / item["file"]
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        observed_rows = sum(1 for _ in reader)
    supplement_checks[item["file"]] = (
        item["file"] in expected_supplements
        and len(header) == item["columns"]
        and observed_rows == item["rows"]
        and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    )
if supplement_manifest.get("status") != "pass" or len(supplement_checks) != 26:
    errors.append("supplementary-table manifest contract mismatch")
errors.extend(f"supplementary-table check failed: {name}" for name, passed in supplement_checks.items() if not passed)

forbidden_suffixes = {
    ".doc", ".docx", ".pdf", ".tex", ".png", ".svg", ".tif", ".tiff",
    ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".npy", ".npz", ".db",
    ".sqlite", ".sqlite3", ".h5", ".hdf5", ".safetensors", ".pyc",
}
forbidden_parts = {
    "paper", "manuscript", "manuscript_pdfs", "supplement", "supplementary",
    "rebuttal", "response_letter", "__pycache__", ".venv", ".cache", "logs",
    "weights", "checkpoints",
}
for relative in relatives:
    path = Path(relative)
    if path.suffix.lower() in forbidden_suffixes:
        errors.append(f"forbidden extension: {relative}")
    if any(part.lower() in forbidden_parts for part in path.parts):
        errors.append(f"forbidden path: {relative}")

privacy_patterns = {
    "private user path": re.compile(r"(?i)(?:[A-Z]:\\Users\\|/home/|/Users/)"),
    "local username": re.compile(r"(?i)\bADMIN\b"),
    "email address": re.compile(r"(?i)[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}"),
    "private key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY"),
    "assigned credential": re.compile(r"(?i)(?:api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"]{8,}"),
    "LaTeX manuscript": re.compile(r"\\(?:documentclass|begin\{document\})"),
}
for relative in relatives:
    if relative == "verify_package.py":
        continue
    path = root / relative
    text = path.read_text(encoding="utf-8", errors="ignore")
    for label, pattern in privacy_patterns.items():
        if pattern.search(text):
            errors.append(f"{label}: {relative}")

def rows(relative):
    with (root / relative).open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


round2_root = root / "research_v2/artifacts/analysis/xai_sanity_cliffs_round2_v1"
round2 = json.loads((round2_root / "crem_mean_primary_summary.json").read_text(encoding="utf-8"))
round2_manifest = json.loads((round2_root / "manifest.json").read_text(encoding="utf-8"))
sentinel_root = root / "research_v2/artifacts/analysis/xai_sanity_cliffs_round2_sentinel_v1"
sentinel = json.loads((sentinel_root / "summary.json").read_text(encoding="utf-8"))
strict = json.loads((sentinel_root / "strict_validation.json").read_text(encoding="utf-8"))
with (root / "figure_source_data/figure4c_robustness_checks.csv").open(encoding="utf-8", newline="") as handle:
    robustness = list(csv.DictReader(handle))
current_sentinel = next(row for row in robustness if row["analysis"].startswith("Component-median sentinel"))
ten_permutations = next(row for row in robustness if row["analysis"].startswith("Ten permutations"))
with (root / "figure_source_data/figure2a_model_absolute.csv").open(encoding="utf-8", newline="") as handle:
    figure2a_fields = csv.DictReader(handle).fieldnames
with (root / "figure_source_data/figure2b_generator_null_adjusted.csv").open(encoding="utf-8", newline="") as handle:
    figure2b_fields = csv.DictReader(handle).fieldnames
science_checks = {
    "round2_targets": round2["empirical_null_adjusted_boundary_extended"]["targets"] == 27,
    "round2_components": round2["empirical_null_adjusted_boundary_extended"]["components"] == 617,
    "round2_script_hash": hashlib.sha256((root / "research_v2/scripts/analyze_round2_revision.py").read_bytes()).hexdigest() == round2_manifest["script_sha256"],
    "round2_output_hashes": all(hashlib.sha256((round2_root / name).read_bytes()).hexdigest() == expected for name, expected in round2_manifest["output_sha256"].items()),
    "sentinel_status": sentinel["status"] == "pass",
    "sentinel_targets": sentinel["count_validation"]["observed"]["targets"] == 27,
    "sentinel_eligible": sentinel["alternative_eligible_components"] == 618,
    "sentinel_paired": sentinel["paired_comparison_components"] == 617,
    "sentinel_unpaired": sentinel["unpaired_alternative_components"] == 1,
    "sentinel_alternative_rows": rows(sentinel_root.relative_to(root).as_posix() + "/alternative_component_configuration_metrics.csv") == 7416,
    "sentinel_paired_rows": rows(sentinel_root.relative_to(root).as_posix() + "/paired_selection_comparison.csv") == 7404,
    "sentinel_target_rows": rows(sentinel_root.relative_to(root).as_posix() + "/target_estimates.csv") == 27,
    "sentinel_bootstrap_rows": rows(sentinel_root.relative_to(root).as_posix() + "/bootstrap.csv") == 5000,
    "sentinel_strict": strict["status"] == "pass" and strict["checks_passed"] == strict["checks_total"] == 49,
    "current_crem_mean_sentinel_estimate": float(current_sentinel["estimate"]) == 0.0,
    "current_crem_mean_sentinel_interval": abs(float(current_sentinel["ci95_low"]) + 0.008450751315270703) < 1e-15 and float(current_sentinel["ci95_high"]) == 0.0,
    "paired_sentinel_selection_change": sentinel["target_balanced_paired_selection_change"] == 0.0 and sentinel["paired_selection_change_ci95"] == [0.0, 0.0],
    "ten_permutation_crem_mean_only": abs(float(ten_permutations["estimate"]) + 0.013502912477144299) < 1e-15,
    "figure2_aggregation_contract": "aggregation" in figure2a_fields and "aggregation" in figure2b_fields,
    "crem_symmetry_molecule_input": hashlib.sha256((root / "research_v2/artifacts/experiment/xai_sanity_cliffs_full_v1/prepared/molecules.csv").read_bytes()).hexdigest() == "003de70aabc2a1218b76bccfba7263b1e22999481dbddcdf39825ac1b6dba3b4",
    "crem_symmetry_generator_script": hashlib.sha256((root / "research_v2/scripts/generate_counterfactuals.py").read_bytes()).hexdigest() == "87dbf0b4ac497fa75edddf5793d6577937260b9a7913d8c6770212d531d9ea88",
}
reporting_root = root / "research_v2/artifacts/analysis/xai_sanity_cliffs_reporting_v1"
reporting_manifest = json.loads((reporting_root / "manifest.json").read_text())
reporting_summary = json.loads((reporting_root / "summary.json").read_text())
science_checks.update({
    "reporting_script_hash": hashlib.sha256((root / "research_v2/scripts/analyze_reporting_sensitivities.py").read_bytes()).hexdigest() == reporting_manifest["script_sha256"],
    "reporting_input_hashes": all(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected for name, expected in reporting_manifest["input_sha256"].items()),
    "reporting_output_hashes": all(hashlib.sha256((reporting_root / name).read_bytes()).hexdigest() == expected for name, expected in reporting_manifest["output_sha256"].items()),
    "reporting_dataset_target_units": reporting_summary["datasets"] == 27 and reporting_summary["distinct_target_ids"] == 26,
    "reporting_primary_unchanged": abs(reporting_summary["primary_unadjusted_unchanged"] + 0.00523271980315315) < 1e-14,
    "reporting_rank_summary_rows": rows(reporting_root.relative_to(root).as_posix() + "/map_similarity_summary.csv") == 7,
    "reporting_sentinel_summary_rows": rows(reporting_root.relative_to(root).as_posix() + "/sentinel_selection_distribution.csv") == 21,
    "reporting_assay_sensitivity_rows": rows(reporting_root.relative_to(root).as_posix() + "/assay_leave_one_out.csv") == 4,
})
errors.extend(f"science smoke failed: {name}" for name, passed in science_checks.items() if not passed)

if errors:
    raise SystemExit("FAIL\n" + "\n".join(errors))
print(f"PASS: {len(entries)} files, {payload_bytes} bytes, 16 figure sources, 26 supplementary tables, {len(science_checks)} science checks")
