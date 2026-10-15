# Response-target diagnostic extension

This archive supplies the 9 October 2026 computational extension. Extract into a new folder and run commands from that folder. Preserve the directory structure. Original project code retains its MIT license; MoleculeACE material retains the bundled upstream notices; WOMBAT source structures are CC0-1.0. Protocol and evaluation files describe the pre-integration execution state and are preserved verbatim. Historical document-preservation checks are retained as records, not rerun against a bundled obsolete manuscript.

## Environment and frozen-cache replay

The executed Python and package versions are in research_v2/artifacts/analysis/deep_construct_20261009_v1/runtime.json. Install requirements_extension.txt into a separate environment. No GPU or new model training is required for these commands:

```
python verify_archive.py
python research_v2/scripts/analyze_deep_construct.py development
python research_v2/scripts/analyze_deep_construct.py external
python research_v2/scripts/diagnose_deep_construct.py
python research_v2/scripts/plot_deep_construct.py
```

Verify the archive before replay, because analysis replaces derived files. The two analyze commands rebuild prevalence-normalized AP, within-size response references and the 5,000-draw intervals from retained mutants; development replay also checks the original oracle medians. Diagnose rebuilds support/tie separation, native-accounting summaries and scaffold sensitivity. Plot rebuilds Figure 5. Timing fields and renderer metadata may differ. The archive-root manifest identifies the delivered files; the older nested manifest also lists the full source parquet and the historical document-specific verifier, which are not bundled here. The source parquet is downloaded separately.

## Independent source and generator rebuild

The selected external list, native reaction traces and exact frozen CReM database are included. In a fresh extraction, `python research_v2/scripts/check_deep_construct_native.py` compares eight upstream and optimized generation cases. To regenerate all selected external mutants, run `python research_v2/scripts/generate_deep_construct_external.py`, then repeat the analysis commands. The original run took about 17 minutes on its recorded host; this package QA replays frozen inputs rather than claiming a new full generator run.

To repeat selection, `python research_v2/scripts/prepare_deep_construct.py` downloads the approximately 158 MB pinned parquet, verifies its hash, excludes all 30 supplied benchmark tables and writes the selected list. The pinned URL and SHA-256 are in input_provenance.json. Run these generation/selection commands only in a separate extraction, because they replace derived cache files there. The 256-source selection and eligibility rules, internal amendments and failed/unsupported case policy are fixed in protocol.md.

## Interpretation

CSV fields called removal/removed represent allowed-region concept occupancy; insertion represents an accounting residual. They are intentionally retained historical field names. Native true_removed/true_inserted fields come from product atom origins. Read interpretation_amendment.md before interpreting these as chemical events. Occupancy is a label-aligned diagnostic, not a deployable explainer. The external panel is motif-conditioned, has no canonical overlap with MoleculeACE but has some scaffold overlap, and has no independent bioactivity labels.

The public activity-cliff base remains at tag rereview-20261009; this extension is supplied here as Additional file 3, not claimed to be part of that tag.
