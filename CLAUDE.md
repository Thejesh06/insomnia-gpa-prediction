# CLAUDE.md — Project Context

## Project Overview

**Theme:** Impact of Insomnia on Academic Performance  
**Type:** Research replication + original data collection + ML classification + Streamlit web app  
**Goal:** Replicate a published research paper on student sleep and GPA, collect primary survey data from our own university, and build a web app that predicts GPA class (Good/Excellent vs Below) from a student's sleep and lifestyle inputs.

---

## Datasets

| File | Description |
|------|-------------|
| `data/Student Insomnia and Educational Outcomes Dataset_version-2.csv` | Base paper's dataset — 15 survey questions, used for replication |
| `data/survey_dataset.csv` | Our university's primary data — 13 questions, Department column unique to this dataset |

**Pipeline is run per-dataset:**
```
python src/full_pipeline/run_pipeline.py base
python src/full_pipeline/run_pipeline.py survey
```

**Important:** All scripts receive the **raw CSV path** via `sys.argv[1]` and do their own encoding. The preprocessed CSVs (`outputs/processed/`) are saved by `run_pipeline.py` but not consumed by analysis scripts — they serve as reference artifacts and will be used by the Streamlit app.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.12 |
| Data | pandas, numpy |
| ML | scikit-learn (LR, DT, KNN, RF, NB, SVM), imbalanced-learn (SMOTE) |
| Stats | scipy (pearsonr, ttest_ind, kruskal), statsmodels (Logit) |
| Visualisation | matplotlib, seaborn |
| App (planned) | Streamlit |
| Pipeline | subprocess-based (`run_pipeline.py`) |

---

## File Structure

```
Mini_project/
├── CLAUDE.md                          ← this file
├── data/
│   ├── Student Insomnia...v2.csv      ← base paper dataset (15 Qs)
│   └── survey_dataset.csv             ← university survey dataset (13 Qs)
│
├── src/
│   ├── preprocessing/
│   │   └── data_preprocessing.py      ← load_and_preprocess() — used by pipeline + app
│   │
│   ├── analysis/
│   │   ├── demographic_analysis.py    ← Table 1a/1b/1c: gender, year, dept distributions
│   │   ├── behavioral_analysis.py     ← Table 2: correlation, regression, mediation, moderation
│   │   ├── table4_grpgpa.py           ← Table 4: t-test + Kruskal + Cohen's d (high vs low GPA)
│   │   └── year_trends_table3.py      ← Table 3 + 3b: GPA/sleep/stress trends by year & dept
│   │
│   ├── visualization/
│   │   ├── fig2_correlation_heatmap.py ← Fig 2: Pearson correlation heatmap (all features)
│   │   ├── feature_imp.py              ← Fig 5: SVM permutation feature importance
│   │   ├── insomnia-analysis.py        ← Fig 1a/1b/1c: insomnia by gender/year/dept [PENDING RENAME]
│   │   └── radar_plots.py              ← Fig 6a/6b: radar charts (High vs Low GPA, Male vs Female)
│   │
│   ├── models/
│   │   ├── table5_LR.py                ← Table 5: Logistic Regression with OR, CI, p-values (base dataset only)
│   │   ├── table6_unbalanced_models.py ← Table 6: 6-model CV comparison, no SMOTE (baseline)
│   │   ├── table7_smote.py             ← Table 7: MAIN MODEL — 6-model CV + SMOTE (replicates base paper)
│   │   └── table8_smote_corrected.py   ← Table 8: Corrected SMOTE (train-only) — our methodological improvement
│   │
│   ├── clustering/
│   │   ├── kmeans_clustering.py        ← Fig 3: K-means (3 clusters) on 12 behavioral features
│   │   └── pca_t-sne.py                ← Fig 4a/4b: PCA + t-SNE projections [PENDING RENAME to pca_tsne.py]
│   │
│   └── full_pipeline/
│       └── run_pipeline.py             ← Orchestrator: preprocesses then runs all scripts in order
│
├── outputs/
│   ├── processed/
│   │   ├── base_clean.csv              ← preprocessed base dataset (reference artifact)
│   │   └── survey_clean.csv            ← preprocessed survey dataset (used by app)
│   ├── figures/                        ← all generated plots (12 files)
│   └── tables/                         ← all generated CSV tables (10 files)
│
└── App/                                ← Streamlit app (TO BE BUILT)
    ├── app.py                          ← main Streamlit app (pending)
    └── model/
        └── smote_model.pkl             ← saved trained model (pending)
```

---

## Encoding Design (Important)

Scripts read **raw CSV** and encode themselves. There are two encoding groups:

**Group 1 — Core 7-feature scripts** (table6, table7, table8, feature_imp, radar_plots, clustering):
- Use `freq_map` (combined short + long form), `sleep_map`, `stress_map`
- Features: `sleep_onset`, `sleep_duration`, `night_arousals`, `screen_time`, `caffeine`, `exercise`, `stress`
- Target: binary (1 = Good/Excellent GPA, 0 = otherwise)

**Group 2 — table5_LR.py** (base dataset only, 12 features):
- Uses separate per-column maps: `freq_map` (long-form only), `simple_map`, `duration_map`, `quality_map`, `class_map`, `deadline_map`, `caf_map`, `stress_map`
- Extra features: `sleep_quality`, `difficulty_concentrating`, `fatigue`, `miss_classes`, `miss_deadlines`
- Hard-coded column rename dict — only works on base paper dataset (intentional replication)

**Group 3 — behavioral_analysis.py**:
- Uses ordinal `gpa_map` (1–5 scale) instead of binary target — needed for regression analysis

**`data_preprocessing.py`** uses lowercase keys (normalises text first) — used only by `run_pipeline.py` and will be used by the Streamlit app for encoding live user input.

**Bug fixed across all scripts:** `"5-6 hours"` was missing from all `sleep_map` dicts (survey dataset has this option). Added as value `2` (same bucket as 4-5 hours, below healthy threshold).

---

## Model Comparison Story (Resume Narrative)

| Table | Script | Method | Purpose |
|-------|--------|---------|---------|
| Table 5 | `table5_LR.py` | Logistic Regression + statsmodels | Statistical inference (OR, CI, p-values) |
| Table 6 | `table6_unbalanced_models.py` | 6-model StratifiedKFold CV, no SMOTE | Baseline — shows class imbalance problem |
| Table 7 | `table7_smote.py` | 6-model CV + SMOTE on full data | **Main model** — replicates base paper exactly |
| Table 8 | `table8_smote_corrected.py` | 6-model + SMOTE on train only | Our correction — avoids data leakage |

**Narrative:** We replicated the base paper's methodology (Table 7) and achieved matching results, then identified and addressed a SMOTE data leakage issue in the original approach (Table 8). Tables 6 and 5 provide baseline and statistical context.

---

## Completed Refactoring

| File | Status |
|------|--------|
| `data_preprocessing.py` | Done |
| `table5_LR.py` | Done |
| `table6_unbalanced_models.py` | Done |
| `table7_smote.py` | Done |
| `smote_train.py` → `table8_smote_corrected.py` | Done (renamed + cleaned) |
| `run_pipeline.py` | Done |
| `demographic_analysis.py` | Done |
| `behavioral_analysis.py` | Done |
| `table4_grpgpa.py` | Done |
| `year_trends_table3.py` | Done |
| `fig2_correlation_heatmap.py` | Pending |
| `feature_imp.py` | Pending |
| `insomnia-analysis.py` → `insomnia_analysis.py` | Pending (rename required) |
| `radar_plots.py` | Pending |
| `kmeans_clustering.py` | Pending |
| `pca_t-sne.py` → `pca_tsne.py` | Pending (rename required) |
| `App/app.py` (Streamlit) | Pending (to be built) |

---

## Pending File Renames

Two filenames have hyphens which make them invalid Python module names:
- `src/visualization/insomnia-analysis.py` → `insomnia_analysis.py`
- `src/clustering/pca_t-sne.py` → `pca_tsne.py`

`run_pipeline.py` has already been updated to reference the new names.

---

## App Plan (Streamlit)

**Dataset:** Survey dataset only (questions match app input form exactly)  
**Model:** `table7_smote.py` trained on survey data → saved as `App/model/smote_model.pkl`  
**Encoding:** `data_preprocessing.py`'s `load_and_preprocess()` logic reused for live input

**App flow:**
1. User fills 7 dropdowns (survey question answer options)
2. Input encoded using same maps as training
3. StandardScaler applied (fitted on survey training data, saved alongside model)
4. Model predicts GPA class + confidence
5. Feature importance panel shows which factors most impacted the result

**Survey input fields for app:**
- Difficulty Falling Asleep: Never / Rarely (1-2×/wk) / Sometimes (3-4×/wk) / Often (5-6×/wk) / Every night
- Average Sleep Hours: Less than 4h / 4-5h / 5-6h / 6-7h / 7-8h / More than 8h
- Waking at Night: Never / Rarely (1-2×/wk) / Sometimes (3-4×/wk) / Often (5-6×/wk) / Every night
- Screen Time Before Bed: Never / Rarely (1-2×/wk) / Sometimes (3-4×/wk) / Often (5-6×/wk) / Every night
- Caffeine Use: Never / Rarely (1-2×/wk) / Sometimes (3-4×/wk) / Often (5-6×/wk) / Every day
- Physical Activity: Never / Rarely (1-2×/wk) / Sometimes (3-4×/wk) / Often (5-6×/wk) / Every day
- Stress Level: No stress / Low stress / High stress / Extremely high stress

---

## Key Decisions Made

1. **Keep both datasets** — base for replication, survey for app. Output folders to be split into `outputs/base/` and `outputs/survey/` (planned).
2. **`table8_smote_corrected.py`** added as our methodological contribution over the base paper.
3. **`smote_train.py` deleted** — replaced by `table8_smote_corrected.py`.
4. **`table5_LR.py` is base-dataset only** — intentional, it replicates the paper's LR analysis which uses 12 features not present in the survey.
5. **`plt.show()` removed from all scripts** — prevents blocking in server/app environments.
6. **All random seeds set to 42** — reproducibility. `table7_smote.py` had `random_state=43` for RandomForest — corrected to 42.
7. **App uses survey-trained model** — survey questions align with the input form exactly.
