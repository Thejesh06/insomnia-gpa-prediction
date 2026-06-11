import pandas as pd
import numpy as np
import sys
import os

from scipy.stats import ttest_ind, kruskal


# ----------------------------
# Load data
# ----------------------------

data_path    = sys.argv[1]
dataset_type = sys.argv[2]
df = pd.read_csv(data_path)

if "Timestamp" in df.columns:
    df = df.drop(columns=["Timestamp"])


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
sleep_duration_col = get_col(["sleep hours", "average"])
night_arousals_col = get_col(["waking", "wake"])
sleep_quality_col  = get_col(["sleep quality", "overall quality"])
concentration_col  = get_col(["concentration", "concentrating"])
fatigue_col        = get_col(["fatigue", "fatigued"])
miss_classes_col   = get_col(["skip classes", "classes missed"])
miss_deadlines_col = get_col(["deadlines", "assignment"])
screen_col         = get_col(["screen", "electronic"])
caffeine_col       = get_col(["caffeine"])
exercise_col       = get_col(["exercise", "physical activity"])
stress_col         = get_col(["stress"])
gpa_col            = get_col(["gpa", "academic performance"])


# ----------------------------
# Encoding maps
# ----------------------------

freq_map = {
    "Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4, "Always": 5,
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

sleep_quality_map = {
    "Very poor": 1, "Poor": 2, "Average": 3, "Good": 4, "Very good": 5,
}

# Miss classes uses per-month scale in base dataset, per-week short form in survey
miss_class_map = {
    "Never": 1,
    "Rarely (1-2 times a month)": 2,
    "Sometimes (1-2 times a week)": 3,
    "Often (3-4 times a week)": 4,
    "Always": 5,
    "Rarely": 2,
    "Sometimes": 3,
    "Often": 4,
}

impact_map = {
    "No impact": 1, "Minor impact": 2, "Moderate impact": 3,
    "Major impact": 4, "Severe impact": 5,
}

stress_map = {
    "No stress": 1, "Low stress": 2, "High stress": 3, "Extremely high stress": 4,
}

gpa_map = {
    "Poor": 1, "Below Average": 2, "Average": 3, "Good": 4, "Excellent": 5,
}


# ----------------------------
# Apply encoding
# ----------------------------

df["Sleep_Onset"]              = df[sleep_onset_col].map(freq_map)        if sleep_onset_col    else np.nan
df["Sleep_Duration"]           = df[sleep_duration_col].map(sleep_map)    if sleep_duration_col else np.nan
df["Night_Arousals"]           = df[night_arousals_col].map(freq_map)     if night_arousals_col else np.nan
df["Sleep_Quality"]            = df[sleep_quality_col].map(sleep_quality_map) if sleep_quality_col else np.nan
df["Difficulty_Concentrating"] = df[concentration_col].map(freq_map)      if concentration_col  else np.nan
df["Fatigue"]                  = df[fatigue_col].map(freq_map)             if fatigue_col        else np.nan
df["Miss_Classes"]             = df[miss_classes_col].map(miss_class_map) if miss_classes_col   else np.nan
df["Miss_Deadlines"]           = df[miss_deadlines_col].map(impact_map)   if miss_deadlines_col else np.nan
df["Screen_Time"]              = df[screen_col].map(freq_map)              if screen_col         else np.nan
df["Caffeine"]                 = df[caffeine_col].map(freq_map)            if caffeine_col       else np.nan
df["Exercise"]                 = df[exercise_col].map(freq_map)            if exercise_col       else np.nan
df["Stress"]                   = df[stress_col].map(stress_map)            if stress_col         else np.nan

df["GPA_score"] = df[gpa_col].map(gpa_map) if gpa_col else np.nan
df["High_GPA"]  = df[gpa_col].apply(lambda x: 1 if x in ["Good", "Excellent"] else 0) if gpa_col else 0

df = df.dropna(subset=["GPA_score"])

high = df[df["High_GPA"] == 1]
low  = df[df["High_GPA"] == 0]


# ----------------------------
# Statistical comparison functions
# ----------------------------

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    pooled_std = np.sqrt(((nx - 1) * np.var(x) + (ny - 1) * np.var(y)) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std != 0 else 0


def effect_label(d):
    d = abs(d)
    if d < 0.2:   return "Negligible"
    elif d < 0.5: return "Small"
    elif d < 0.8: return "Medium"
    else:         return "Large"


def sig_label(p):
    if pd.isna(p):  return "NA"
    elif p < 0.001: return "***"
    elif p < 0.01:  return "**"
    elif p < 0.05:  return "*"
    else:           return "ns"


# ----------------------------
# Compute Table 4
# ----------------------------

variables = [
    "Sleep_Onset", "Sleep_Duration", "Night_Arousals", "Sleep_Quality",
    "Difficulty_Concentrating", "Fatigue", "Miss_Classes", "Miss_Deadlines",
    "Screen_Time", "Caffeine", "Exercise", "Stress",
]

results = []

for var in variables:
    x = high[var].dropna()
    y = low[var].dropna()

    if len(x) < 2 or len(y) < 2:
        t_p, kw_p, d = np.nan, np.nan, np.nan
    else:
        t_p  = ttest_ind(x, y, equal_var=False).pvalue
        kw_p = kruskal(x, y).pvalue
        d    = cohens_d(x, y)

    results.append({
        "Variable":    var,
        "T-test p":    t_p,
        "T-test sig":  sig_label(t_p),
        "Cohen d":     round(d, 3) if not pd.isna(d) else np.nan,
        "Effect size": effect_label(d) if not pd.isna(d) else "NA",
        "KW p":        kw_p,
        "KW sig":      sig_label(kw_p),
    })

table4 = pd.DataFrame(results)

print("\nTable 4: GPA Group Differences (High vs Low)\n")
print(f"High GPA: {len(high)}  |  Low GPA: {len(low)}\n")
print(table4.to_string(index=False))

os.makedirs(f"outputs/{dataset_type}/tables", exist_ok=True)
table4.to_csv(f"outputs/{dataset_type}/tables/table4_result.csv", index=False)
