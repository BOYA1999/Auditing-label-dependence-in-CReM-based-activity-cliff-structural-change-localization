# Auditing label dependence in CReM structural-change localization

This checkout contains the 15 October 2026 minor revision: an exact two-thirds calibration gate; a shared-endpoint conditional reference for the complete nested statistic; seed-level paired mapping sensitivity; and the response-target extension with common-population native-removal comparisons and traceable molecular examples. Tables S1–S46 and their scientific source records are supplied. The previous fixed code snapshot remains at tag `rereview-20261009`; its unchanged frozen inputs remain attached to `rereview-20261008`.

## Restore and verify

Use Python 3.12 and the pinned packages in `requirements_minor_revision.txt`, which includes the original dependency files. CairoSVG also requires the Cairo rendering runtime. Run from the repository root:

```
python -B fetch_core_inputs.py
python -B verify_package.py
python -B verify_archive.py
```

The first command restores four large frozen inputs and checks their exact hashes. The package validator checks current inventory and hashes, supplementary CSVs, figure sources and explicitly permitted scientific figures. The archive validator checks the 88 original response-target payload files preserved from Additional file 3. That extension's original README and protocols describe its historical package state; the current checkout now includes its code and caches.

## Replay the new analyses

```
python -B research_v2/scripts/recompute_minor_threshold_20261015.py
python -B research_v2/scripts/analyze_minor_nested_mapping.py
python -B research_v2/scripts/verify_minor_nested_mapping.py
python -B research_v2/scripts/analyze_minor_extension.py
python -B research_v2/scripts/plot_minor_extension.py
```

These commands recompute frozen-prediction and frozen-score statistics without fitting models. The gate replay retains the original seed aggregation and checks four actual boundary cases and all 5,000 historical bootstrap draws. Mapping sensitivity shares each candidate across label conditions, seeds and configurations. The endpoint-label reference recomputes the nested statistic for 2,000 assignments; it is a conditional reference, not a population interval or a training-label exchangeability test. The extension replays native reaction origins for all 8,598 retained mutants and recomputes 5,000-draw common-population summaries. Two small source-origin inputs support this replay without downloading the full source parquet.

Run inventory/hash validation before replay. Analyses regenerate derived files in dedicated minor-revision directories. Numerical CSVs and scientific assertions provide replay checks; runtime fields, environment records, nested manifests and rendered-figure metadata can change and are not scientific discrepancy tests. Original frozen result directories remain. Current S7/S8 replace the former 82-cell presentation, preserved in Git history.

## Results and scope

The exact gate retains 86/162 cells: historical pooled-null contrast −0.00743340, 95% interval [−0.02165214, −0.00135780]. The principal 617-pair raw point (−0.00523272) and endpoint-exact point (−0.00699105) are unchanged. The complete-statistic conditional reference has a nonzero center: row-level expectation centering alone does not center nested medians. Six unreproduced mappings remain explicit. S43/S44 report the conditional reference and paired mapping sensitivity; S45/S46 report the common-population response comparison and source/example identities.

S1–S10 and S12–S26 retain historical diagnostics, with corrected S7/S8 eligibility; adjusted quantities there use the historical 500-assignment pooled null. S11 uses assessed mapping denominators. S27–S34 contain exact-null and matched controls, and S35/S36 contain six-case exclusion. S37–S42 are the original response-target extension. Distinct estimands and populations must not be substituted for one another.

The extension retains 256 selected structures, 251 supported structures, 536 outside-index traces and 15 partial-region replacements. The selected WOMBAT source panel comprises 237 ordinary negatives and 19 similarity-selected negatives; none matches the source or validation Pattern 1 SMARTS. This is a source-conditioned panel, not a motif-positive sample, and it has no independent bioactivity labels. Occupancy is an intentionally label-aligned diagnostic. Findings concern this CReM replacement/assignment setting and do not establish general failure of molecular XAI.

## Earlier replays and source rebuild

Earlier commands remain: `replay_review_core.py`, `replay_rereview.py`, and `analyze_count_oracle_revision.py --oracle-only` under `research_v2/scripts`. `README_REPLAY.md` gives original extension replays and optional source-selection/native-generation commands. Use a separate extraction for selection or generation because they replace caches. The roughly 158 MB pinned WOMBAT parquet is downloaded separately; the exact frozen training-derived CReM database is included.

This is frozen-statistics analysis, not a complete model refit or independent biological validation. Full refitting still requires documented pretrained assets, representations and fitted-model resources. No manuscript, author declarations, DOI or journal submission is included or claimed. The existing MIT license and upstream MoleculeACE notices remain unchanged; see `DATABASE_PROVENANCE.md`.

The [core-input archive](https://github.com/BOYA1999/Auditing-label-dependence-in-CReM-based-activity-cliff-structural-change-localization/releases/download/rereview-20261008/xai_sanity_cliffs_core_inputs_20261008.zip) contains 91,554,490 bytes. `CORE_INPUTS_MANIFEST.json` records extracted hashes; `package_manifest.json` records the archive hash.
