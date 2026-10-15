# Response-target revision

This analysis supplements the existing response-target extension with a common-supported-population comparison, a replay of every saved native reaction and two traceable molecular illustrations. It preserves the original 256-structure analysis and uses 251 supported structures only for the explicitly matched heteroatom comparison. The corresponding halogen population contains 110 structures. Native elemental removal is not extended to aromaticity.

From the repository root, with the original Additional file 3 directory structure restored:

```text
python -B research_v2/scripts/analyze_minor_extension.py
python -B research_v2/scripts/plot_minor_extension.py
```

The original extension supplies selected parents, retained mutants, reaction patterns, native atom origins and existing analysis code. The two small files in `inputs` supply the audited upstream source memberships and source-pool totals. The analysis verifies the membership checksum and joins memberships to the exact selected source strings; it also recomputes both Pattern 1 SMARTS matches. A complete source-selection audit can additionally run the original `prepare_deep_construct.py` in a separate extraction; it downloads and verifies the pinned source parquet. That full selection replay was performed for this revision and returned the identical 256-structure list.

The executed environment uses Python 3.12, RDKit 2026.3.3, NumPy 1.26.4, pandas 2.3.3, matplotlib 3.10.8, CairoSVG 2.8.2 and Pillow 11.3.0. CairoSVG requires its Cairo rendering runtime. No GPU or trained model is needed. The analysis reconstructs every saved reaction transformation; it does not regenerate the complete CReM candidate neighbourhoods or retrain models.

`table_s45.csv` reports marginal medians and paired contrasts separately. `table_s46.csv` reports the upstream source categories and example identities. Atom scores, region responses, bootstrap draws, all 8,598 reaction checks, outside-index records, partial-region records and the full example atom provenance accompany these tables. `figure5_source.json` is the numeric and structural source for the updated figure; PNG/TIFF are 300 dpi and PDF/SVG retain vectors. The original Figure 5 data remain in the original extension.

The 256 selected structures consist of 237 ordinary negatives and 19 similarity-selected negatives in the upstream Pattern 1 source pool; none matches the source or validation SMARTS. The source pool itself contains positives and both negative categories. The panel is therefore source-conditioned, not a motif-positive sample. Source structures retain the upstream CC0-1.0 declaration; the existing code licence and third-party notices remain applicable. The two upstream code snapshots used for the local audit are not required for this replay or included in the minimal public payload.
