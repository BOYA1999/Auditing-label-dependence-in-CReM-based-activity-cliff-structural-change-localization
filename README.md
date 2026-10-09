# Activity-cliff localization rereview revision 2026-10-08

Code and frozen statistical records corresponding to the 2026-10-08 rereview manuscript and Tables S1-S36, published on 9 October 2026. The fixed snapshot is release tag `rereview-20261008`. Commit `3322b470e5537bd14bbaf661f148683f53bf78e9` remains in the history and predates these revisions. No DOI or journal submission is claimed.

## Restore and replay

Clone this repository at tag `rereview-20261008`, or download the source archive for that release. Use Python 3.12 and the pinned requirements. Run the following commands from the repository root. The first command downloads the companion frozen inputs from the release, verifies its SHA-256 and restores the four original input paths:

```
python -B fetch_core_inputs.py
python -B verify_package.py
python -B research_v2/scripts/replay_review_core.py
python -B research_v2/scripts/replay_rereview.py
python -B research_v2/scripts/analyze_count_oracle_revision.py --oracle-only
```

The validator checks inventory, file hashes, path boundaries, 16 figure-source CSVs, 36 supplementary-table CSVs and the four restored inputs. The core replay recomputes exact-null statistics, generator controls and mapping assessment from restored atom scores, pairs, masks and scored mutants. The rereview replay filters the six unreproduced mapping keys, recomputes 5000-draw intervals and matched model/generator summaries, and compares every rereview output hash. Exact-count localization is replayed on the original development mutants without training. Learned-model mutant errors are supplied as frozen outputs; redoing those requires the omitted saved models and representation arrays.

## Result and provenance layers

The raw principal point remains -0.00523271980315315. The full-panel exact endpoint contrast is -0.0069910516781531915. The corrected mapping audit assesses 611/617 pairs and finds 77 ambiguous pairs; six failures remain explicit. Rereview exclusion leaves the raw and endpoint point estimates unchanged. The endpoint exclusion interval is [-0.0154841863219514, -0.0018927868361068067]. All six failures also belong to the strict-unmatched population, so its exclusion population contains 575 pairs.

The revision analysis directory contains frozen model/generator metrics and exact-null, map-stability and oracle results. The rereview directory adds mapping_status_by_dataset.csv, explicit stored/reproduced atom and bond records, deleted rows, per-dataset results, primary/bootstrap summaries, matched absolute values and paired trained-minus-generator differences. Atom indices are zero based; each common-bond row is [left_bond_id, right_bond_id, left_begin, left_end, right_begin, right_end]. Unreproduced common-atom/bond fields are empty lists and are never supplied as successful mappings.

Tables S1-S10 and S12-S26 retain historical diagnostics; adjusted values there use the historical 500-assignment pooled null. S11 uses assessed mapping denominators. S27-S34 contain the exact-null and matched-control revision. S35-S36 contain six-case exclusion results. Full and exclusion intervals use distinct bootstrap realizations; the original raw interval remains historical. See the manuscript for estimand-specific interpretation.

## Limits and public access

This is frozen-statistics replay, not a complete model refit or independent biological validation. Full refitting still requires documented pretrained assets, representation arrays, CReM databases and fitted-model resources. The companion restores the previously missing atom scores (57498330 bytes), predictions (188988140 bytes) and initial prepared caches. Input provenance files, companion input hashes and MANIFEST.sha256 identify exact bytes. Original MIT code licence and upstream MoleculeACE terms are preserved.

No manuscript or author declarations are included. HISTORICAL_README.md and historical_scripts document the older package. The fixed public revision and companion input archive are available without repository authentication. Author-supplied declarations and journal submission remain separate author tasks.

## Frozen input download

[Core-input archive](https://github.com/BOYA1999/Auditing-label-dependence-in-CReM-based-activity-cliff-structural-change-localization/releases/download/rereview-20261008/xai_sanity_cliffs_core_inputs_20261008.zip) (91554490 bytes). SHA-256: `f113d4f5a55f4dcd0ffd5a5f31cf9667071b7d9b7950cc8a8fb688e4ef63d5ec`. See `CORE_INPUTS_MANIFEST.json` for the four extracted file hashes. The large inputs are release assets rather than Git-tracked CSVs.
