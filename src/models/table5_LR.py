import pandas as pd
import numpy as np
import sys
import os

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


# ----------------------------
# Load data
# ----------------------------

data_path    = sys.argv[1]
dataset_type = sys.argv[2]
df = pd.read_csv(data_path)


# ----------------------------
# Rename columns (base dataset only)
# ----------------------------

df = df.rename(columns={
    "3. How often do you have difficulty falling asleep at night? ":          "sleep_onset",
    "4. On average, how many hours of sleep do you get on a typical day?":    "sleep_duration",
    "5. How often do you wake up during the night and have trouble falling back asleep?": "night_arousals",
    "6. How would you rate the overall quality of your sleep?":               "sleep_quality",
    "7. How often do you experience difficulty concentrating during lectures or studying due to lack of sleep?": "difficulty_concentrating",
    "8. How often do you feel fatigued during the day, affecting your ability to study or attend classes?": "fatigue",
    "9. How often do you miss or skip classes due to sleep-related issues (e.g., insomnia, feeling tired)?": "miss_classes",
    "10. How would you describe the impact of insufficient sleep on your ability to complete assignments and meet deadlines?": "miss_deadlines",
    "11. How often do you use electronic devices (e.g., phone, computer) before going to sleep?": "screen_time",
    "12. How often do you consume caffeine (coffee, energy drinks) to stay awake or alert?":     "caffeine",
    "13. How often do you engage in physical activity or exercise?":          "exercise",
    "14. How would you describe your stress levels related to academic workload?":               "stress",
    "15. How would you rate your overall academic performance (GPA or grades) in the past semester?": "gpa_category",
})


# ----------------------------
# Encoding maps
# ----------------------------

# Sleep onset, night arousals, screen time
freq_map = {
    "Never": 1,
    "Rarely (1-2 times a week)": 2,
    "Sometimes (3-4 times a week)": 3,
    "Often (5-6 times a week)": 4,
    "Every night": 5,
}

# Concentration, fatigue (short-form scale)
simple_map = {
    "Never": 1,
    "Rarely": 2,
    "Sometimes": 3,
    "Often": 4,
    "Always": 5,
}

duration_map = {
    "Less than 4 hours": 1,
    "4-5 hours": 2,
    "6-7 hours": 3,
    "7-8 hours": 4,
    "More than 8 hours": 5,
}

quality_map = {
    "Very poor": 1,
    "Poor": 2,
    "Average": 3,
    "Good": 4,
    "Very good": 5,
}

# Miss classes uses per-month scale, not per-week
class_map = {
    "Never": 1,
    "Rarely (1-2 times a month)": 2,
    "Sometimes (1-2 times a week)": 3,
    "Often (3-4 times a week)": 4,
    "Always": 5,
}

deadline_map = {
    "No impact": 1,
    "Minor impact": 2,
    "Moderate impact": 3,
    "Major impact": 4,
    "Severe impact": 5,
}

caf_map = {
    "Never": 1,
    "Rarely (1-2 times a week)": 2,
    "Sometimes (3-4 times a week)": 3,
    "Often (5-6 times a week)": 4,
    "Every day": 5,
}

stress_map = {
    "No stress": 1,
    "Low stress": 2,
    "High stress": 3,
    "Extremely high stress": 4,
}


# ----------------------------
# Apply encoding
# ----------------------------

df["sleep_onset"]              = df["sleep_onset"].map(freq_map)
df["sleep_duration"]           = df["sleep_duration"].map(duration_map)
df["night_arousals"]           = df["night_arousals"].map(freq_map)
df["sleep_quality"]            = df["sleep_quality"].map(quality_map)
df["difficulty_concentrating"] = df["difficulty_concentrating"].map(simple_map)
df["fatigue"]                  = df["fatigue"].map(simple_map)
df["miss_classes"]             = df["miss_classes"].map(class_map)
df["miss_deadlines"]           = df["miss_deadlines"].map(deadline_map)
df["screen_time"]              = df["screen_time"].map(freq_map)
df["caffeine"]                 = df["caffeine"].map(caf_map)
df["exercise"]                 = df["exercise"].map(caf_map)
df["stress"]                   = df["stress"].map(stress_map)


# ----------------------------
# Target variable
# ----------------------------

df["target"] = df["gpa_category"].apply(
    lambda x: 1 if x in ["Good", "Excellent"] else 0
)

df = df.drop(columns=["gpa_category"]).dropna()


# ----------------------------
# Features
# ----------------------------

features = [
    "sleep_onset", "sleep_duration", "night_arousals", "sleep_quality",
    "difficulty_concentrating", "fatigue", "miss_classes", "miss_deadlines",
    "screen_time", "caffeine", "exercise", "stress",
]

X = df[features]
y = df["target"]


# ----------------------------
# Standardise (z-score)
# ----------------------------

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


# ----------------------------
# Logistic regression (statsmodels)
# ----------------------------

X_const = sm.add_constant(X_scaled)
model = sm.Logit(y, X_const).fit()


# ----------------------------
# Results table
# ----------------------------

params = model.params
conf   = model.conf_int()
pvals  = model.pvalues

results = pd.DataFrame({
    "Variable":   params.index,
    "Odds Ratio": np.exp(params),
    "CI Lower":   np.exp(conf[0]),
    "CI Upper":   np.exp(conf[1]),
    "p-value":    pvals,
}).drop("const").round(3)

print("\nTable 5: Logistic Regression Results\n")
print(results.to_string(index=False))

os.makedirs(f"outputs/{dataset_type}/tables", exist_ok=True)
results.to_csv(f"outputs/{dataset_type}/tables/table5_LR.csv", index=False)
