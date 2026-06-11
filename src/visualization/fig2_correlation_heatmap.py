import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os


# ----------------------------
# Load data
# ----------------------------

data_path    = sys.argv[1]
dataset_type = sys.argv[2]
df = pd.read_csv(data_path)


# ----------------------------
# Column detection
# ----------------------------

def get_col(names):
    for col in df.columns:
        for n in names:
            if n.lower() in col.lower():
                return col
    return None


sleep_onset    = get_col(["difficulty falling asleep"])
sleep_duration = get_col(["sleep hours", "average"])
night          = get_col(["waking", "wake"])
sleep_quality  = get_col(["sleep quality", "overall"])
concentration  = get_col(["concentrating", "concentration"])
fatigue        = get_col(["fatigue"])
miss_classes   = get_col(["skip classes", "classes missed"])
miss_deadlines = get_col(["deadline", "assignment"])
screen         = get_col(["screen", "electronic"])
caffeine       = get_col(["caffeine"])
exercise       = get_col(["exercise", "physical activity"])
stress         = get_col(["stress"])
gpa            = get_col(["gpa", "academic performance"])


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

quality_map = {
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
# Build feature dataframe
# ----------------------------

data = pd.DataFrame()
data["Sleep_Onset"]              = df[sleep_onset].map(freq_map)      if sleep_onset    else 0
data["Sleep_Duration"]           = df[sleep_duration].map(sleep_map)  if sleep_duration else 0
data["Night_Arousals"]           = df[night].map(freq_map)             if night          else 0
data["Sleep_Quality"]            = df[sleep_quality].map(quality_map) if sleep_quality  else 0
data["Difficulty_Concentrating"] = df[concentration].map(freq_map)    if concentration  else 0
data["Fatigue"]                  = df[fatigue].map(freq_map)           if fatigue        else 0
data["Miss_Classes"]             = df[miss_classes].map(miss_class_map) if miss_classes  else 0
data["Miss_Deadlines"]           = df[miss_deadlines].map(impact_map) if miss_deadlines else 0
data["Screen_Time"]              = df[screen].map(freq_map)            if screen         else 0
data["Caffeine"]                 = df[caffeine].map(freq_map)          if caffeine       else 0
data["Exercise"]                 = df[exercise].map(freq_map)          if exercise       else 0
data["Stress"]                   = df[stress].map(stress_map)          if stress         else 0
data["GPA"]                      = df[gpa].map(gpa_map)                if gpa            else 0

data = data.apply(lambda col: col.fillna(col.median()))


# ----------------------------
# Correlation heatmap
# ----------------------------

corr_matrix = data.corr()

plt.figure(figsize=(11, 9))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
)
plt.title("Pearson Correlation Matrix — Sleep, Behavioural, and Academic Variables", pad=12)
plt.tight_layout()

os.makedirs(f"outputs/{dataset_type}/figures", exist_ok=True)
plt.savefig(f"outputs/{dataset_type}/figures/fig2_heatmap.jpg", dpi=300)
plt.close()

print("Fig 2: Correlation heatmap saved.")
