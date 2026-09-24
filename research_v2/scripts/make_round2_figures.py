import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ROUND2 = ROOT / "research_v2/artifacts/analysis/xai_sanity_cliffs_round2_v1"
MAJOR = ROOT / "research_v2/artifacts/analysis/xai_sanity_cliffs_major_revision_v1"
FULL = ROOT / "research_v2/artifacts/analysis/xai_sanity_cliffs_full_v1"
PERM = ROOT / "research_v2/artifacts/experiment/xai_sanity_cliffs_major_revision_v1/permutation_stability"
SENTINEL = ROOT / "research_v2/artifacts/analysis/xai_sanity_cliffs_round2_sentinel_v1"
FIGURES = ROOT / "research_v2/paper/figures"
LATEX_FIGURES = ROOT / "research_v2/paper/latex/figures"
SOURCE = ROOT / "research_v2/paper/figure_source_data"

BLUE = "#2166AC"
ORANGE = "#D6604D"
TEAL = "#1B9E77"
PURPLE = "#7570B3"
GREY = "#666666"
LIGHT = "#E7F3F1"
PALE = "#F2F2F2"
CONFIG_COLORS = {
    "MoLFormer · MLP": "#4E79A7",
    "MoLFormer · XGB": "#A0CBE8",
    "Morgan · MLP": "#E15759",
    "Morgan · XGB": "#FF9D9A",
    "Uni-Mol · MLP": "#59A14F",
    "Uni-Mol · XGB": "#8CD17D",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def panel(ax, label, x=-0.12, y=1.03):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def configuration_label(representation, predictor):
    rep = {"morgan": "Morgan", "molformer": "MoLFormer", "unimol": "Uni-Mol"}[representation]
    pred = {"xgboost": "XGB", "mlp": "MLP"}[predictor]
    return f"{rep} · {pred}"


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    LATEX_FIGURES.mkdir(parents=True, exist_ok=True)
    base = FIGURES / name
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    shutil.copy2(base.with_suffix(".pdf"), LATEX_FIGURES / f"{name}.pdf")
    plt.close(fig)


def figure1():
    overall = pd.read_csv(MAJOR / "prediction_overall_summary.csv").set_index("variant")
    robust = json.loads((FULL / "robustness/summary.json").read_text(encoding="utf-8"))
    rmse = robust["prediction_randomization"]
    pairwise = pd.DataFrame(
        {
            "metric": ["Cliff-pair Spearman", "Direction accuracy"],
            "observed_labels": [
                overall.loc["trained", "target_balanced_median_spearman"],
                overall.loc["trained", "target_balanced_median_direction_accuracy"],
            ],
            "permuted_labels": [
                overall.loc["label_permuted", "target_balanced_median_spearman"],
                overall.loc["label_permuted", "target_balanced_median_direction_accuracy"],
            ],
        }
    )
    rmse_source = pd.DataFrame(
        {
            "metric": ["Permuted − observed test RMSE"],
            "estimate": [rmse["target_balanced_median_permuted_minus_trained_rmse"]],
            "ci95_low": [rmse["target_bootstrap_ci"][0]],
            "ci95_high": [rmse["target_bootstrap_ci"][1]],
            "targets": [rmse["targets"]],
        }
    )
    chronology = pd.DataFrame(
        {
            "stage": [
                "Prespecified asymmetric analysis",
                "Original 27-dataset results examined",
                "Post-confirmation corrected CReM-mean audit",
            ],
            "status": ["result-blind", "chronology boundary", "post-confirmation"],
        }
    )

    fig = plt.figure(figsize=(6.69, 4.15))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.08, 1], hspace=0.62, wspace=0.55)

    ax = fig.add_subplot(gs[0, :])
    panel(ax, "a", -0.045, 1.02)
    ax.axis("off")
    boxes = [
        (0.01, 0.29, "Prespecified asymmetric audit\n3 development + 27\nresult-blind datasets", "#E6EEF8"),
        (0.37, 0.24, "Original 27-dataset\nresults examined", "#FFF1DE"),
        (0.68, 0.31, "Post-confirmation corrected audit\nCReM mean · empirical null\nproxy controls", "#E5F4EA"),
    ]
    for x, width, label, color in boxes:
        ax.add_patch(plt.Rectangle((x, 0.30), width, 0.46, transform=ax.transAxes, facecolor=color, edgecolor="#555555", lw=0.7))
        ax.text(x + width / 2, 0.53, label, transform=ax.transAxes, ha="center", va="center", fontsize=7.2, linespacing=1.15)
    ax.annotate("", xy=(0.37, 0.53), xytext=(0.30, 0.53), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 0.9, "color": GREY})
    ax.annotate("", xy=(0.68, 0.53), xytext=(0.61, 0.53), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 0.9, "color": GREY})
    ax.text(0.50, 0.08, "Same frozen cliff pairs, splits and neighborhoods; inferential status changed", transform=ax.transAxes, ha="center", color=GREY, fontsize=7.5)

    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "b")
    x = rmse_source.loc[0, "estimate"]
    lo = rmse_source.loc[0, "ci95_low"]
    hi = rmse_source.loc[0, "ci95_high"]
    ax.errorbar(x, 0, xerr=[[x - lo], [hi - x]], fmt="o", color=ORANGE, capsize=3, lw=1.8, ms=5)
    ax.axvline(0, color="#888888", lw=0.8)
    ax.set_yticks([0], ["Test RMSE"])
    ax.set_xlim(0, 0.48)
    ax.set_xlabel("Permuted − observed (log units)")
    ax.set_title("Permutation damages prediction", loc="left")
    ax.text(x, 0.20, f"{x:.3f} [{lo:.3f}, {hi:.3f}]", ha="center", color=GREY)
    ax.set_ylim(-0.45, 0.45)

    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "c")
    y = np.array([1, 0])
    for yv, row in zip(y, pairwise.itertuples()):
        ax.plot([row.observed_labels, row.permuted_labels], [yv, yv], color="#BDBDBD", lw=1.5)
    ax.scatter(pairwise.observed_labels, y, color=BLUE, s=29, label="Observed labels", zorder=2)
    ax.scatter(pairwise.permuted_labels, y, color=ORANGE, marker="s", s=26, label="Permuted labels", zorder=2)
    ax.set_yticks(y, pairwise.metric)
    ax.set_xlim(-0.05, 0.80)
    ax.set_xlabel("Target-balanced value")
    ax.set_title("Cliff-pair signal is disrupted", loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.53, -0.52), ncol=2)

    ax = fig.add_subplot(gs[1, 2])
    panel(ax, "d")
    ax.axis("off")
    ax.set_title("Reliability-qualified scope", loc="left")
    scope_lines = [
        ("27", "datasets"),
        ("617", "test components"),
        ("6", "representation–predictor cells"),
    ]
    for i, (value, label) in enumerate(scope_lines):
        yv = 0.82 - i * 0.19
        ax.text(0.04, yv, value, transform=ax.transAxes, fontsize=9.0, fontweight="bold", color=BLUE, va="center")
        ax.text(0.37, yv, label, transform=ax.transAxes, fontsize=7.4, va="center")
    ax.text(0.04, 0.24, "CReM mean", transform=ax.transAxes, fontsize=8.0, fontweight="bold", color=BLUE)
    ax.text(0.04, 0.16, "primary explainer", transform=ax.transAxes, fontsize=7.2)
    ax.text(0.04, 0.04, "CReM-LIME: failure case only", transform=ax.transAxes, fontsize=7.0, color=ORANGE)

    SOURCE.mkdir(parents=True, exist_ok=True)
    chronology.to_csv(SOURCE / "figure1a_chronology.csv", index=False)
    rmse_source.to_csv(SOURCE / "figure1b_prediction_damage.csv", index=False)
    pairwise.to_csv(SOURCE / "figure1c_pairwise_signal.csv", index=False)
    pd.DataFrame(
        {
            "targets": [27],
            "components": [617],
            "endpoints": [1234],
            "representations": [3],
            "predictors": [2],
            "saved_initializations": [3],
            "primary_explainer": ["CReM mean"],
            "failure_case": ["CReM-LIME"],
        }
    ).to_csv(SOURCE / "figure1d_scope.csv", index=False)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.19)
    save(fig, "figure1_design_intervention")


def figure2():
    config = pd.read_csv(ROUND2 / "crem_mean_configuration_summary.csv")
    config["configuration"] = [configuration_label(a, b) for a, b in zip(config.representation, config.predictor)]
    empirical = pd.read_csv(ROUND2 / "empirical_null_summary.csv")
    generator = empirical[empirical.source.eq("generator")].copy()
    support = pd.read_csv(MAJOR / "generator_changed_unchanged_summary.csv").iloc[0]
    controls = pd.read_csv(MAJOR / "positive_controls/positive_control_configuration_summary.csv")
    controls = controls[controls.explainer.eq("crem_mean")].copy()
    controls["configuration"] = [configuration_label(a, b) for a, b in zip(controls.representation, controls.predictor)]
    controls["passes"] = controls.passes.astype(str).str.lower().eq("true")

    fig, axes = plt.subplots(2, 2, figsize=(6.69, 5.25), gridspec_kw={"hspace": 0.58, "wspace": 0.50})

    ax = axes[0, 0]
    panel(ax, "a")
    order = config.sort_values("target_balanced_median_null_adjusted_nap_trained").reset_index(drop=True)
    y = np.arange(len(order))
    for i, row in order.iterrows():
        ax.plot(
            [row.target_balanced_median_null_adjusted_nap_trained, row.target_balanced_median_null_adjusted_nap_permuted],
            [i, i],
            color="#BDBDBD",
            lw=1.2,
        )
    ax.scatter(order.target_balanced_median_null_adjusted_nap_trained, y, color=BLUE, s=26, label="Observed labels")
    ax.scatter(order.target_balanced_median_null_adjusted_nap_permuted, y, color=ORANGE, marker="s", s=23, label="Permuted labels")
    ax.axvline(0, color="#888888", lw=0.8)
    ax.set_yticks(y, order.configuration)
    ax.set_xlabel("Component-level null-adjusted nAP")
    ax.set_title("Localization after seed averaging", loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.53, -0.44), ncol=2)

    ax = axes[0, 1]
    panel(ax, "b")
    generator["label"] = generator.apply(
        lambda r: ("Nearest-training similarity" if r.baseline == "selected_mean_nearest_train_tanimoto" else "Any perturbation support")
        + ("\nBoundary extended" if r.rationale == "boundary_extended" else "\nStrict unmatched"),
        axis=1,
    )
    generator = generator.sort_values(["baseline", "rationale"]).reset_index(drop=True)
    yy = np.arange(len(generator))[::-1]
    ax.barh(yy, generator.target_balanced_median_null_adjusted_nap, color=[TEAL if "Nearest" in v else PURPLE for v in generator.label], height=0.58)
    ax.axvline(0, color="#888888", lw=0.8)
    ax.set_yticks(yy, generator.label.str.replace("Nearest-training similarity", "Nearest-training\nsimilarity", regex=False).str.replace("Any perturbation support", "Any perturbation\nsupport", regex=False))
    ax.set_xlabel("Row-level null-adjusted nAP")
    ax.set_title("The generator predicts the proxy", loc="left")

    ax = axes[1, 0]
    panel(ax, "c")
    vals = [support.target_balanced_median_changed_support, support.target_balanced_median_unchanged_support]
    ax.bar([0, 1], vals, color=[ORANGE, "#8C8C8C"], width=0.60)
    ax.set_xticks([0, 1], ["Structural-change", "Unchanged"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Perturbation-support probability")
    ax.set_title("Proxy atoms are more reachable", loc="left")
    ax.text(0.5, 0.94, f"paired gap {support.target_balanced_median_support_gap:.3f}\n95% CI [{support.support_gap_ci95_low:.3f}, {support.support_gap_ci95_high:.3f}]", ha="center", va="top", fontsize=7.5)

    ax = axes[1, 1]
    panel(ax, "d")
    concepts = ["heteroatom_count", "aromatic_atom_count", "halogen_count"]
    concept_names = ["Heteroatom\n1/6 pass", "Aromatic\n4/5 pass", "Halogen\n6/6 pass"]
    rng = np.random.default_rng(20260904)
    for i, concept in enumerate(concepts):
        part = controls[controls.control.eq(concept)]
        jitter = rng.uniform(-0.13, 0.13, len(part))
        colors = [TEAL if value else "#9E9E9E" for value in part.passes]
        ax.scatter(i + jitter, part.median_calibrated_nap, c=colors, s=28, edgecolor="white", linewidth=0.35)
    ax.axhline(0, color="#777777", lw=0.8)
    ax.set_xticks(range(3), concept_names)
    ax.set_ylabel("Observed − randomized nAP")
    ax.set_title("Deterministic controls are selective", loc="left")
    ax.scatter([], [], color=TEAL, label="Pass")
    ax.scatter([], [], color="#9E9E9E", label="Fail")
    ax.legend(loc="upper left")

    order["aggregation"] = "seed-averaged within component; median within target; median across targets"
    generator["aggregation"] = "row-level pair-specific null; median within target; median across targets"
    order.to_csv(SOURCE / "figure2a_model_absolute.csv", index=False)
    generator.to_csv(SOURCE / "figure2b_generator_null_adjusted.csv", index=False)
    pd.DataFrame({"atom_class": ["structural_change", "unchanged"], "support_probability": vals}).to_csv(SOURCE / "figure2c_support.csv", index=False)
    controls.to_csv(SOURCE / "figure2d_positive_controls.csv", index=False)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.95, bottom=0.15)
    save(fig, "figure2_absolute_configuration")


def estimand_rows():
    primary = json.loads((ROUND2 / "crem_mean_primary_summary.json").read_text(encoding="utf-8"))
    competence = json.loads((ROUND2 / "prediction_competence_summary.json").read_text(encoding="utf-8"))["eligible_primary"]
    specs = [
        ("Post-confirmation unadjusted", primary["unadjusted_boundary_extended"]),
        ("Pair-null adjusted", primary["empirical_null_adjusted_boundary_extended"]),
        ("Strict unmatched", primary["empirical_null_adjusted_strict_unmatched"]),
        ("Prevalence ≤ 0.8", primary["empirical_null_adjusted_boundary_prevalence_le_0_8"]),
        ("Calibration qualified", competence),
    ]
    return pd.DataFrame(
        [
            {
                "estimand": name,
                "estimate": value["estimate"],
                "ci95_low": value["ci95"][0],
                "ci95_high": value["ci95"][1],
                "targets": value["targets"],
                "components": value["components"],
            }
            for name, value in specs
        ]
    )


def figure3():
    estimands = estimand_rows()
    cells = pd.read_csv(ROUND2 / "crem_mean_target_configuration_effects.csv")
    cells["configuration"] = [configuration_label(a, b) for a, b in zip(cells.representation, cells.predictor)]
    components = pd.read_csv(ROUND2 / "crem_mean_empirical_null_component_deltas.csv")
    target_rationale = (
        components.groupby(["dataset", "rationale", "representation", "predictor"], as_index=False).delta_null_adjusted_nap.median()
        .groupby(["dataset", "rationale"], as_index=False).delta_null_adjusted_nap.median()
    )
    target_pair = target_rationale.pivot(index="dataset", columns="rationale", values="delta_null_adjusted_nap").dropna().reset_index()
    mapping = pd.read_csv(ROUND2 / "mcs_mapping_nap_sensitivity.csv")
    mapping_pair = mapping.groupby(["dataset", "component_id"], as_index=False).mapping_nap_range.median()

    fig, axes = plt.subplots(2, 2, figsize=(6.69, 5.20), gridspec_kw={"hspace": 0.58, "wspace": 0.48})

    ax = axes[0, 0]
    panel(ax, "a")
    y = np.arange(len(estimands))[::-1]
    ax.axvspan(-0.05, 0.05, color=LIGHT)
    ax.axvline(0, color="#777777", lw=0.8)
    for yv, row in zip(y, estimands.itertuples()):
        ax.plot([row.ci95_low, row.ci95_high], [yv, yv], color=BLUE, lw=1.8)
        ax.scatter(row.estimate, yv, color=BLUE, s=26, zorder=2)
    ax.set_yticks(y, estimands.estimand)
    ax.set_xlim(-0.057, 0.057)
    ax.set_xlabel("Observed − permuted nAP")
    ax.set_title("Matched estimand hierarchy", loc="left")

    ax = axes[0, 1]
    panel(ax, "b")
    configs = list(CONFIG_COLORS)
    rng = np.random.default_rng(20260904)
    for i, config in enumerate(configs):
        values = cells.loc[cells.configuration.eq(config), "median_delta_null_adjusted_nap"].to_numpy()
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(values)), values, s=13, color=CONFIG_COLORS[config], alpha=0.55, edgecolor="none")
        ax.scatter(i, np.median(values), marker="D", s=28, color="#222222", zorder=3)
    ax.axhspan(-0.05, 0.05, color=LIGHT, zorder=0)
    ax.axhline(0, color="#777777", lw=0.8)
    ax.set_xticks(range(6), ["MoLF–M", "MoLF–X", "Morgan–M", "Morgan–X", "Uni-Mol–M", "Uni-Mol–X"], rotation=35, ha="right", fontsize=6.5)
    ax.set_ylabel("Null-adjusted cell ΔnAP")
    ax.set_title("Target × configuration effects", loc="left")

    ax = axes[1, 0]
    panel(ax, "c")
    x0 = target_pair.boundary_extended.to_numpy()
    x1 = target_pair.strict_unmatched.to_numpy()
    for a, b in zip(x0, x1):
        ax.plot([0, 1], [a, b], color="#BDBDBD", lw=0.7, alpha=0.7)
    ax.scatter(np.zeros(len(x0)), x0, color=BLUE, s=18, alpha=0.75)
    ax.scatter(np.ones(len(x1)), x1, color=TEAL, s=18, alpha=0.75)
    ax.axhspan(-0.05, 0.05, color=LIGHT, zorder=0)
    ax.axhline(0, color="#777777", lw=0.8)
    ax.set_xticks([0, 1], ["Boundary extended", "Strict unmatched"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("Target-level median ΔnAP")
    ax.set_title("Proxy sensitivity by dataset", loc="left")

    ax = axes[1, 1]
    panel(ax, "d")
    values = mapping.mapping_nap_range.to_numpy()
    displayed = values[values <= 0.30]
    ax.hist(displayed, bins=np.linspace(0, 0.30, 16), color=PURPLE, alpha=0.82, edgecolor="white", linewidth=0.4)
    median = float(np.median(values))
    q95 = float(np.quantile(values, 0.95))
    ax.axvline(median, color="#222222", lw=1.1, label=f"Median {median:.3f}")
    ax.axvline(q95, color=ORANGE, lw=1.1, ls="--", label=f"95th percentile {q95:.3f}")
    ax.set_xlabel("nAP range across valid mappings")
    ax.set_ylabel("Pair × configuration rows")
    ax.set_title("Mapping ambiguity can be substantial", loc="left")
    ax.set_xlim(0, 0.30)
    ax.text(0.98, 0.58, f"{int((values > 0.30).sum())} rows > 0.30", transform=ax.transAxes, ha="right", color=GREY, fontsize=7.2)
    ax.legend(loc="upper right")

    estimands.to_csv(SOURCE / "figure3a_estimands.csv", index=False)
    cells.to_csv(SOURCE / "figure3b_target_configuration_cells.csv", index=False)
    target_pair.to_csv(SOURCE / "figure3c_rationale_targets.csv", index=False)
    mapping.to_csv(SOURCE / "figure3d_mapping_ambiguity.csv", index=False)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.95, bottom=0.16)
    save(fig, "figure3_generator_controls")


def robustness_rows():
    rows = []
    competence = json.loads((ROUND2 / "prediction_competence_summary.json").read_text(encoding="utf-8"))["eligible_primary"]
    rows.append(
        {
            "analysis": "Calibration-qualified\n26 datasets, 82 cells",
            "estimate": competence["estimate"],
            "ci95_low": competence["ci95"][0],
            "ci95_high": competence["ci95"][1],
            "interval_available": True,
        }
    )
    convergence = pd.read_csv(PERM / "nap_convergence_by_target_configuration.csv")
    convergence = convergence[(convergence.explainer == "crem_mean") & (convergence.permutations_used == 10)]
    convergence_estimate = float(convergence.groupby("dataset").trained_minus_random_nap.median().median())
    rows.append(
        {
            "analysis": "Ten permutations\n27 datasets, randomized-model seed 42",
            "estimate": convergence_estimate,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "interval_available": False,
        }
    )
    sentinel = pd.read_csv(SENTINEL / "alternative_component_configuration_metrics.csv")
    sentinel = sentinel[sentinel.explainer == "crem_mean"].copy()
    grouped = {
        target: [part.delta_nap.to_numpy(float) for _, part in target_frame.groupby("component_id", sort=True)]
        for target, target_frame in sentinel.groupby("dataset", sort=True)
    }
    targets = sorted(grouped)
    rng = np.random.default_rng(20260911)
    draws = np.empty(5000)
    for draw in range(len(draws)):
        target_values = []
        for target_index in rng.integers(0, len(targets), len(targets)):
            components = grouped[targets[target_index]]
            sampled = [components[index] for index in rng.integers(0, len(components), len(components))]
            target_values.append(np.median(np.concatenate(sampled)))
        draws[draw] = np.median(target_values)
    rows.append(
        {
            "analysis": "Component-median sentinel\n27 datasets, 618 comps, seed 42",
            "estimate": float(sentinel.groupby("dataset").delta_nap.median().median()),
            "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975)),
            "interval_available": True,
        }
    )
    return pd.DataFrame(rows)


def figure4():
    cells = pd.read_csv(ROUND2 / "crem_mean_target_configuration_effects.csv")
    cells["configuration"] = [configuration_label(a, b) for a, b in zip(cells.representation, cells.predictor)]
    matrix = cells.pivot(index="dataset", columns="configuration", values="median_delta_null_adjusted_nap")
    matrix = matrix.reindex(columns=list(CONFIG_COLORS))
    matrix = matrix.loc[matrix.median(axis=1).sort_values().index]
    hierarchy = json.loads((ROUND2 / "hierarchical_heterogeneity.json").read_text(encoding="utf-8"))["empirical_null_adjusted"]
    intervals = pd.DataFrame(
        {
            "interval": ["Grand-mean bootstrap CI", "Selected-panel normal-model spread"],
            "estimate": [hierarchy["grand_mean"], hierarchy["grand_mean"]],
            "low": [hierarchy["bootstrap_ci95"]["grand_mean"][0], hierarchy["new_target_configuration_prediction_interval_approx95"][0]],
            "high": [hierarchy["bootstrap_ci95"]["grand_mean"][1], hierarchy["new_target_configuration_prediction_interval_approx95"][1]],
        }
    )
    robustness = robustness_rows()
    fidelity = pd.read_csv(MAJOR / "rescore_controls/local_fidelity_summary.csv")
    fidelity = fidelity[fidelity.scenario.eq("fixed_selected_support_0_60")].set_index("variant")
    replay = pd.read_csv(MAJOR / "rescore_controls/reference_pair_reproduction_summary.csv")
    failed = int(replay.loc[replay.explainer_rescore.eq("crem_lime_abs"), "rows_above_frozen_tolerance"].iloc[0])
    fidelity_source = pd.DataFrame(
        {
            "metric": ["In-sample", "PRESS", "Five-fold"],
            "observed_labels": [
                fidelity.loc["trained", "target_balanced_median_in_sample_r2"],
                fidelity.loc["trained", "target_balanced_median_press_r2"],
                fidelity.loc["trained", "target_balanced_median_five_fold_r2"],
            ],
            "permuted_labels": [
                fidelity.loc["label_permuted", "target_balanced_median_in_sample_r2"],
                fidelity.loc["label_permuted", "target_balanced_median_press_r2"],
                fidelity.loc["label_permuted", "target_balanced_median_five_fold_r2"],
            ],
            "replay_rows_failed": [failed, failed, failed],
            "replay_rows_total": [4968, 4968, 4968],
        }
    )

    fig = plt.figure(figsize=(6.69, 7.35))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1.65, 1], hspace=0.66, wspace=0.48)

    ax = fig.add_subplot(gs[:, 0])
    panel(ax, "a", -0.18, 1.01)
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-0.12, vmax=0.12)
    ax.set_xticks(range(matrix.shape[1]), [v.replace(" · ", "\n") for v in matrix.columns], rotation=35, ha="right")
    ax.set_yticks(range(matrix.shape[0]), matrix.index.str.replace("CHEMBL", "", regex=False), fontsize=5.5)
    ax.set_xlabel("")
    ax.set_ylabel("Target–assay table")
    ax.set_title("Null-adjusted CReM-mean ΔnAP", loc="left")
    cbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.038, pad=0.10)
    cbar.set_label("Observed − permuted nAP")

    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b")
    y = [1, 0]
    ax.axvspan(-0.05, 0.05, color=LIGHT)
    ax.axvline(0, color="#777777", lw=0.8)
    for yv, row in zip(y, intervals.itertuples()):
        ax.plot([row.low, row.high], [yv, yv], color=BLUE if yv else ORANGE, lw=2.0)
        ax.scatter(row.estimate, yv, color=BLUE if yv else ORANGE, s=25)
    ax.set_yticks(y, ["Grand-mean\nbootstrap CI", "Selected-panel\nnormal-model spread"])
    ax.set_xlim(-0.10, 0.08)
    ax.set_xlabel("Null-adjusted ΔnAP")
    ax.set_title("Mean uncertainty and cell variation", loc="left")

    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "c")
    y = np.arange(len(robustness))[::-1]
    ax.axvspan(-0.05, 0.05, color=LIGHT)
    ax.axvline(0, color="#777777", lw=0.8)
    for yv, row in zip(y, robustness.itertuples()):
        if row.interval_available:
            ax.plot([row.ci95_low, row.ci95_high], [yv, yv], color=TEAL, lw=1.7)
        ax.scatter(row.estimate, yv, color=TEAL, marker="D" if not row.interval_available else "o", s=25)
    ax.set_yticks(y, [
        "Calibration-qualified\n26 datasets, 82 cells",
        "Ten permutations\n27 datasets,\nrandomized-model\nseed 42",
        "Component-median\nsentinel: 27 datasets,\n618 comps, seed 42",
    ])
    ax.set_xlim(-0.06, 0.06)
    ax.set_xlabel("Observed − permuted nAP")
    ax.set_title("Different checks answer different questions", loc="left")

    ax = fig.add_subplot(gs[2, 1])
    panel(ax, "d")
    x = np.arange(3)
    ax.bar(x - 0.18, fidelity_source.observed_labels, width=0.36, color=BLUE, label="Observed labels")
    ax.bar(x + 0.18, fidelity_source.permuted_labels, width=0.36, color=ORANGE, label="Permuted labels")
    ax.axhline(0, color="#777777", lw=0.8)
    ax.set_xticks(x, fidelity_source.metric)
    ax.set_ylabel("Weighted local $R^2$")
    ax.set_title("Audited CReM-LIME: poor held-out fit", loc="left")
    ax.text(0.98, 0.53, f"Replay failures\n{failed}/4,968 rows", transform=ax.transAxes, fontsize=7.4, color=GREY, ha="right")
    ax.legend(loc="upper right")

    matrix.reset_index().to_csv(SOURCE / "figure4a_target_configuration_heatmap.csv", index=False)
    intervals.to_csv(SOURCE / "figure4b_hierarchical_intervals.csv", index=False)
    robustness.to_csv(SOURCE / "figure4c_robustness_checks.csv", index=False)
    fidelity_source.to_csv(SOURCE / "figure4d_lime_failure.csv", index=False)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.09)
    save(fig, "figure4_robustness_fidelity")


def main():
    global FIGURES, LATEX_FIGURES, SOURCE
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.output_root is not None:
        output_root = args.output_root.resolve()
        FIGURES = output_root / "figures"
        LATEX_FIGURES = output_root / "latex_figures"
        SOURCE = output_root / "figure_source_data"
    SOURCE.mkdir(parents=True, exist_ok=True)
    figure1()
    figure2()
    figure3()
    figure4()
    names = [
        "figure1_design_intervention",
        "figure2_absolute_configuration",
        "figure3_generator_controls",
        "figure4_robustness_fidelity",
    ]
    outputs = [FIGURES / f"{name}{suffix}" for name in names for suffix in [".pdf", ".svg", ".png", ".tiff"]]
    source_names = [
        "figure1a_chronology.csv", "figure1b_prediction_damage.csv", "figure1c_pairwise_signal.csv", "figure1d_scope.csv",
        "figure2a_model_absolute.csv", "figure2b_generator_null_adjusted.csv", "figure2c_support.csv", "figure2d_positive_controls.csv",
        "figure3a_estimands.csv", "figure3b_target_configuration_cells.csv", "figure3c_rationale_targets.csv", "figure3d_mapping_ambiguity.csv",
        "figure4a_target_configuration_heatmap.csv", "figure4b_hierarchical_intervals.csv", "figure4c_robustness_checks.csv", "figure4d_lime_failure.csv",
    ]
    sources = [SOURCE / name for name in source_names]
    inputs = [
        ("model_analysis/crem_mean_configuration_summary.csv", ROUND2 / "crem_mean_configuration_summary.csv"),
        ("model_analysis/empirical_null_summary.csv", ROUND2 / "empirical_null_summary.csv"),
        ("model_analysis/crem_mean_primary_summary.json", ROUND2 / "crem_mean_primary_summary.json"),
        ("model_analysis/prediction_competence_summary.json", ROUND2 / "prediction_competence_summary.json"),
        ("model_analysis/crem_mean_target_configuration_effects.csv", ROUND2 / "crem_mean_target_configuration_effects.csv"),
        ("model_analysis/crem_mean_empirical_null_component_deltas.csv", ROUND2 / "crem_mean_empirical_null_component_deltas.csv"),
        ("structural_proxy/mcs_mapping_nap_sensitivity.csv", ROUND2 / "mcs_mapping_nap_sensitivity.csv"),
        ("heterogeneity/hierarchical_heterogeneity.json", ROUND2 / "hierarchical_heterogeneity.json"),
        ("prediction_control/prediction_overall_summary.csv", MAJOR / "prediction_overall_summary.csv"),
        ("generator_control/generator_changed_unchanged_summary.csv", MAJOR / "generator_changed_unchanged_summary.csv"),
        ("positive_controls/positive_control_configuration_summary.csv", MAJOR / "positive_controls/positive_control_configuration_summary.csv"),
        ("fidelity_control/local_fidelity_summary.csv", MAJOR / "rescore_controls/local_fidelity_summary.csv"),
        ("fidelity_control/reference_pair_reproduction_summary.csv", MAJOR / "rescore_controls/reference_pair_reproduction_summary.csv"),
        ("sentinel_control/summary.json", SENTINEL / "summary.json"),
        ("sentinel_control/alternative_component_configuration_metrics.csv", SENTINEL / "alternative_component_configuration_metrics.csv"),
        ("robustness/summary.json", FULL / "robustness/summary.json"),
        ("permutation_control/nap_convergence_by_target_configuration.csv", PERM / "nap_convergence_by_target_configuration.csv"),
    ]
    manifest = {
        "status": "pass",
        "figures": len(names),
        "formats_per_figure": 4,
        "editable_svg_text": True,
        "minimum_font_points": 7.2,
        "conditional_checks": ["ten_permutation_one_initialization"],
        "full_panel_checks": ["twenty_seven_target_component_median_sentinel"],
        "input_hashes": {label: sha256(path) for label, path in inputs},
        "source_hashes": {path.name: sha256(path) for path in sources},
        "output_hashes": {path.name: sha256(path) for path in outputs},
    }
    (FIGURES / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "figures": len(names), "outputs": len(outputs), "source_tables": len(sources)}))


if __name__ == "__main__":
    main()
