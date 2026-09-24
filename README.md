# xai-sanity-cliffs-v1

Code and public-data package for a matched label-randomization audit of activity-cliff structural-change localization. The current post-confirmation analysis is restricted to unsigned CReM-mean explanations across three molecular representations (Morgan, MoLFormer and Uni-Mol), two predictors (XGBoost and MLP) and three saved model seeds. CReM-LIME is retained only as a documented numerical and held-out-fidelity failure case.

This repository intentionally contains no manuscript, supplementary manuscript, editorial correspondence, Word file, LaTeX source, manuscript figure, author declaration or submission metadata.

## Current result and scope

Across 27 targets originally held out under the asymmetric protocol, 617 eligible activity-cliff components and six representation-by-predictor configurations, the target-balanced CReM-mean estimate was `-0.005233` before empirical-null adjustment (conditional 95% target-and-component bootstrap interval `[-0.013903, 0.000000]`) and `-0.008317` after the pair-specific empirical-null adjustment (conditional 95% interval `[-0.015768, -0.002769]`). These intervals condition on the saved model initializations, label permutations, neighborhoods and observed-label-selected cliff set.

The full 27-target component-median sentinel analysis retained 618 eligible components. Under the current CReM-mean-only scope, its target-balanced trained-minus-label-permuted delta was `0.000000` (95% target-and-component bootstrap interval `[-0.008451, 0.000000]`). Among the 617 components available under both sentinel selections, the target-balanced selection-change estimate was also `0.000000` (95% interval `[0.000000, 0.000000]`).

The selected-panel descriptive normal-model spread remained broad (`[-0.081368, 0.064139]` after empirical-null adjustment); empirical cell percentiles were `[-0.083688, 0.047074]`. Neither range has validated predictive coverage for unseen configurations. Legacy field names in the immutable statistical outputs are retained for provenance, not as stronger prediction claims.

The principal panel comprises 27 target-assay datasets but 26 distinct target identifiers. Reporting sensitivities distinguish direct map agreement from localization-score contrasts: supported-atom and all-atom matched-seed rank summaries are `0.096181` and `0.767389`, with 848 undefined supported correlations. There is no within-observed-label seed-stability reference or causal map-response claim. In the 328 sentinel components that actually change pair, the absolute-change row median is `0.118386` and P95 is `0.520038`; a zero pooled signed median does not imply local invariance. Omitting either CHEMBL237 assay retains negative contrasts. Five-to-ten ordered permutations change the pooled contrast by `0.004906`, so convergence at five is not established.

## Data provenance

The 30 redistributed assay tables are the public MoleculeACE benchmark at upstream commit `7e6de0bd2968c56589c580f2a397f01c531ede26`, obtained on 1 September 2026. MoleculeACE identifies ChEMBL as the original bioactivity source. This study did not query ChEMBL directly, and the exact historical ChEMBL release, extraction query and access snapshot used to construct the redistributed benchmark are not present upstream and cannot be reconstructed from these tables.

Full provenance, licence boundaries and the ChEMBL database citation are in [`DATABASE_PROVENANCE.md`](DATABASE_PROVENANCE.md). The data contain molecular structures, assay values and benchmark annotations; no patient, clinical, contact, institutional or private data were used.

## Included material

- Frozen protocols and configurations under `research_v2/artifacts/idea_sanity_benchmark/` and `research_v2/configs/`.
- Public MoleculeACE tables and their upstream licence under `external/MoleculeACE/`.
- Preparation records, selected molecular pairs, row-level prediction/explanation inputs and compact derived tables under `research_v2/artifacts/`.
- The complete 22-file round-2 statistical analysis and the complete 11-file 27-target sentinel analysis.
- Machine-readable exports for all 26 supplementary tables, plus their row/column/hash manifest, under `research_v2/artifacts/analysis/xai_sanity_cliffs_supplement_tables_v1/`.
- Reporting-sensitivity summaries, bootstrap draws and input/output hashes under `research_v2/artifacts/analysis/xai_sanity_cliffs_reporting_v1/`.
- Nine compact sentinel execution-evidence files: preparation and cache audits, generator/representation/explanation summaries and row-level pair explanations. Large intermediates are excluded.
- Scripts for preparation, model fitting, explanation scoring and robustness analyses under `research_v2/scripts/`.
- Exactly 16 current, explicitly named tabular figure-source datasets under `figure_source_data/`. Manuscript figures and manuscript text remain excluded.
- Exact direct dependency versions, external-model identifiers and the executed hardware/resource record in `environment.md` and `requirements.txt`.

Pretrained weights, representation arrays, fitted model binaries, CReM databases, per-model work directories, caches and logs are deliberately excluded. These are large or environment-specific intermediates, not private study data.

Some retained artifact paths use immutable run identifiers required by the frozen configurations and hash contracts. Those identifiers have no release-status meaning.
The frozen configuration field and identifier `confirmation_targets` records historical terminology from the original asymmetric protocol; it is retained unchanged only for hash provenance and does not make the current corrected estimand confirmatory.

## Verify and smoke-test the package

From the repository root:

```powershell
python verify_package.py
```

This standard-library-only check verifies every entry in `MANIFEST.sha256`, manifest uniqueness and ordering, package counts, the exact 16-table figure-source allowlist, all 26 supplementary-table exports, prohibited paths/extensions, privacy and credential patterns, and the compact round-2/sentinel/reporting result contracts. Git's local `.git` metadata is excluded from the payload inventory. `.gitattributes` preserves the exact tracked bytes, including line endings, so the same manifest can be checked after cloning on Windows or Linux.

To regenerate the reporting sensitivities from included saved records, run `python -B research_v2/scripts/analyze_reporting_sensitivities.py` from a disposable package copy with NumPy and pandas installed. This reproduces statistical summaries, not model training or independent scientific validation.

## Reproduce the packaged sentinel analysis and figures

The compact package can deterministically regenerate the full 27-target sentinel analysis and the current figures from included inputs. These commands write to paths recorded in the frozen configurations, so run them in a disposable clone if you want to preserve a pristine archive.

```powershell
python research_v2/scripts/analyze_sentinel_selection_sensitivity.py `
  --config research_v2/configs/round2_sentinel_v1.json

python research_v2/scripts/make_round2_figures.py `
  --output-root regenerated/round2
```

The sentinel command regenerates all 11 packaged sentinel-analysis files byte for byte. The figure command writes outside the tracked tree and must regenerate exactly the 16 CSV files in the packaged `figure_source_data/` directory. The repository also retains scripts and compact results for the earlier robustness analyses; `make_round2_figures.py` is the current figure-regeneration entry point.

## Validate the CReM symmetry optimization

With the pinned `crem`, `pandas` and `rdkit` versions in `requirements.txt`, run:

```powershell
python -B research_v2/scripts/validate_crem_symmetry_optimization.py `
  --molecules research_v2/artifacts/experiment/xai_sanity_cliffs_full_v1/prepared/molecules.csv `
  --output regenerated/crem_symmetry_optimization_validation.json
```

The check compares the optimized and reference CReM fragment enumerators on 60 seed-fixed random molecules and six explicitly symmetric molecules. It requires exact fragment-tuple set equality and exact returned-list order equality. The packaged molecule table has SHA-256 `003de70aabc2a1218b76bccfba7263b1e22999481dbddcdf39825ac1b6dba3b4`; the tested generator script has SHA-256 `87dbf0b4ac497fa75edddf5793d6577937260b9a7913d8c6770212d531d9ea88`.

## Full-rerun boundary

The 22 current round-2 analysis outputs are distributed with hashes, but rerunning `analyze_round2_revision.py` additionally requires the unbundled 57,498,330-byte atom-score table and 188,988,140-byte molecule-prediction table. They were excluded from this compact repository together with other large intermediates. With those files restored to the paths in `round2_revision_v1.json`, the command is:

```powershell
python research_v2/scripts/analyze_round2_revision.py `
  --config research_v2/configs/round2_revision_v1.json
```

A complete refit or a fresh 27-target sentinel run further requires the unbundled MoLFormer and Uni-Mol assets, representation arrays, model binaries, CReM database and mutant caches. The full sentinel command is:

```powershell
python research_v2/scripts/run_sentinel_selection_sensitivity.py `
  --config research_v2/configs/round2_sentinel_v1.json
```

The strict model/array validator likewise requires those unbundled intermediates. CUDA, driver, conformer and third-party model revisions can affect regenerated intermediates. The included row-level inputs and hashes provide an exact route for the packaged statistical analyses without claiming bitwise cross-hardware identity.

## Repository and licensing

The code/data package is distributed through [the project repository](https://github.com/BOYA1999/Auditing-label-dependence-in-CReM-based-activity-cliff-structural-change-localization), separate from journal additional files. Use the exact Git commit SHA to identify the version obtained; `MANIFEST.sha256` identifies its payload. No DOI is claimed.

Original project code is covered by the repository's existing MIT `LICENSE`, preserved from the author-provided repository. Bundled MoleculeACE files retain their upstream `external/MoleculeACE/LICENCE`; database provenance and third-party model/dependency terms remain separate. The root licence does not relicense third-party materials.
