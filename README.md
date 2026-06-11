# Impact of Insomnia on Academic Performance

A research replication and original data collection project examining how student sleep patterns affect GPA, combined with a machine learning classification pipeline and an interactive Streamlit web application.

---

## Overview

This project:
- **Replicates** a published research paper on student sleep and academic performance
- **Collects** primary survey data from our university (768 students)
- **Builds** a 6-model ML classification pipeline to predict GPA category
- **Addresses** a methodological flaw (SMOTE data leakage) in the original paper
- **Deploys** a Streamlit web app for live GPA predictions based on sleep and lifestyle inputs

---

## Datasets

| Dataset | Students | Questions | Source |
|---------|----------|-----------|--------|
| Base paper dataset | 996 | 15 | Published research |
| University survey | 768 | 13 + Department | Primary collection |

**Binary target:** Good/Excellent GPA (1) vs Average or Below (0)

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.12 |
| Data | pandas, numpy |
| ML | scikit-learn (LR, DT, KNN, RF, NB, SVM), imbalanced-learn (SMOTE) |
| Stats | scipy, statsmodels |
| Visualisation | matplotlib, seaborn |
| App | Streamlit, Plotly |

---

## Project Structure

```
insomnia-gpa-prediction/
├── data/                          ← Both datasets (raw CSV)
├── src/
│   ├── preprocessing/             ← Data encoding & cleaning
│   ├── analysis/                  ← Demographics, behavioral, GPA group analysis
│   ├── visualization/             ← Correlation heatmap, insomnia plots, radar charts
│   ├── models/                    ← Table 5 (LR), Table 6, Table 7, Table 8
│   ├── clustering/                ← K-means, PCA, t-SNE
│   └── full_pipeline/             ← run_pipeline.py orchestrator
├── outputs/
│   ├── base/figures/ & tables/    ← Base paper results (11 figures, 9 tables)
│   └── survey/figures/ & tables/  ← Survey results (12 figures, 10 tables)
├── notebooks/
│   └── analysis.ipynb             ← Standalone notebook (both datasets)
└── App/
    ├── app.py                     ← Streamlit web app
    └── model/
        ├── train_model.py         ← Model training script
        ├── smote_model.pkl        ← Trained Random Forest
        ├── scaler.pkl             ← Fitted StandardScaler
        └── feature_profiles.pkl  ← Class averages for radar chart
```

---

## Running the Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline on base paper dataset
python src/full_pipeline/run_pipeline.py base

# Run full pipeline on survey dataset
python src/full_pipeline/run_pipeline.py survey
```

---

## Model Comparison Story

| Table | Method | Purpose |
|-------|--------|---------|
| Table 5 | Logistic Regression + statsmodels | Statistical inference (OR, CI, p-values) |
| Table 6 | 6-model CV, no SMOTE | Baseline — shows class imbalance problem |
| Table 7 | 6-model CV + SMOTE on full data | Replicates base paper exactly |
| **Table 8** | **6-model + SMOTE on train only** | **Our fix — avoids data leakage** |

**Key finding:** We replicated the base paper's methodology (Table 7, RF accuracy ~77%) and then identified a SMOTE data leakage issue in the original approach. Table 8 corrects this with SMOTE applied inside the cross-validation loop.

---

## Streamlit App

The app predicts a student's GPA category from 7 sleep and lifestyle inputs:

```bash
# Train the model first
python App/model/train_model.py

# Launch the app
streamlit run App/app.py
```

**App features:**
- GPA prediction with decision threshold tuned for class imbalance
- Plotly gauge chart showing P(Good/Excellent GPA)
- Radar chart: Your Profile vs High GPA Average vs Average/Below Average
- Per-feature risk assessment with personalised tips
- Model feature importance panel

---

## Key Results (Survey Dataset)

| Model | Accuracy | AUC |
|-------|----------|-----|
| Random Forest (Table 7) | 77.4% | **0.864** |
| SVM (Table 7) | 69.2% | 0.763 |
| Decision Tree (Table 7) | 72.7% | 0.728 |
| Random Forest (Table 8 — honest) | 70.8% | 0.701 |

**Top predictors:** Screen time before bed, sleep onset difficulty, night arousals

---

## Features Used

| Feature | Description |
|---------|-------------|
| `sleep_onset` | Difficulty falling asleep (frequency) |
| `sleep_duration` | Average nightly sleep hours |
| `night_arousals` | Frequency of waking at night |
| `screen_time` | Screen use before bed |
| `caffeine` | Caffeine consumption frequency |
| `exercise` | Physical activity frequency |
| `stress` | Academic stress level |
