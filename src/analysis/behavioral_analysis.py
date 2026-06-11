import pandas as pd
import numpy as np
import sys
import os

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


# ----------------------------
# Load data
# ----------------------------

data_path    = sys.argv[1]
dataset_type = sys.argv[2]
df = pd.read_csv(data_path)


# ----------------------------
# Column detection
# ----------------------------

def get_col(possible_names):
    for col in df.columns:
        for name in possible_names:
            if name.lower() in col.lower():
                return col
    return None


sleep_onset_col    = get_col(["difficulty falling asleep"])
night_arousals_col = get_col(["waking", "wake up"])
sleep_duration_col = get_col(["sleep hours", "hours of sleep"])
stress_col         = get_col(["stress"])
exercise_col       = get_col(["physical activity", "exercise"])
gpa_col            = get_col(["academic performance", "gpa"])


# ----------------------------
# Encoding maps
# ----------------------------

freq_map = {
    "Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4, "Always": 5,
    "Every day": 5,
    "Rarely (1-2 times a week)": 2,
    "Sometimes (3-4 times a week)": 3,
    "Often (5-6 times a week)": 4,
    "Every night": 5,
}

sleep_map = {
    "Less than 4 hours": 1,
    "4-5 hours": 2,
    "5-6 hours": 2,
    "6-7 hours": 3,
    "7-8 hours": 4,
    "More than 8 hours": 5,
}

stress_map = {
    "No stress": 1,
    "Low stress": 2,
    "High stress": 3,
    "Extremely high stress": 4,
}

# Ordinal GPA scale used for regression (not binary classification)
gpa_map = {
    "Poor": 1,
    "Below Average": 2,
    "Average": 3,
    "Good": 4,
    "Excellent": 5,
}


# ----------------------------
# Apply encoding
# ----------------------------

if sleep_onset_col:
    df["sleep_onset"] = df[sleep_onset_col].map(freq_map)
if night_arousals_col:
    df["night_arousals"] = df[night_arousals_col].map(freq_map)
if sleep_duration_col:
    df["sleep_duration"] = df[sleep_duration_col].map(sleep_map)
if stress_col:
    df["stress"] = df[stress_col].map(stress_map)
if exercise_col:
    df["exercise"] = df[exercise_col].map(freq_map)
if gpa_col:
    df["gpa"] = df[gpa_col].map(gpa_map)

df = df.dropna(subset=["sleep_onset", "night_arousals", "sleep_duration", "stress", "exercise", "gpa"])


# ----------------------------
# Sleep score composite
# ----------------------------

df["sleep_score"] = df["sleep_onset"] + df["sleep_duration"] + df["night_arousals"]


# ----------------------------
# Insomnia flag (DSM-inspired rule)
# ----------------------------

df["insomnia"] = (
    (df["sleep_onset"] >= 3) &
    (df["night_arousals"] >= 3) &
    (df["sleep_duration"] < 6)
).astype(int)

insomnia_rate = df["insomnia"].mean() * 100


# ----------------------------
# Correlations and regressions
# ----------------------------

corr, _ = pearsonr(df["sleep_score"], df["gpa"])

def r2(X_col, y_col):
    m = LinearRegression()
    m.fit(df[[X_col]], df[y_col])
    return m.score(df[[X_col]], df[y_col])

r2_sleep_gpa    = r2("sleep_score", "gpa")
r2_sleep_stress = r2("sleep_score", "stress")
r2_stress_gpa   = r2("stress", "gpa")

df["interaction"] = df["sleep_score"] * df["exercise"]
mod = LinearRegression()
mod.fit(df[["sleep_score", "exercise", "interaction"]], df["gpa"])
r2_moderation = mod.score(df[["sleep_score", "exercise", "interaction"]], df["gpa"])


# ----------------------------
# Results table
# ----------------------------

table2 = pd.DataFrame({
    "Metric": [
        "Insomnia prevalence (%)",
        "Sleep score -> GPA correlation (r)",
        "Sleep score -> GPA regression (R2)",
        "Sleep -> Stress mediation (R2)",
        "Stress -> GPA mediation (R2)",
        "Sleep x Exercise moderation (R2)",
    ],
    "Value": [
        round(insomnia_rate, 2),
        round(corr, 3),
        round(r2_sleep_gpa, 3),
        round(r2_sleep_stress, 3),
        round(r2_stress_gpa, 3),
        round(r2_moderation, 3),
    ],
})

print("\nTable 2: Behavioral Modeling Results\n")
print(table2.to_string(index=False))

os.makedirs(f"outputs/{dataset_type}/tables", exist_ok=True)
table2.to_csv(f"outputs/{dataset_type}/tables/table2_result.csv", index=False)
