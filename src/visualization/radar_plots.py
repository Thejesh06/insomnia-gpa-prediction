import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
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
gender         = get_col(["gender"])


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


# ----------------------------
# Build feature dataframe
# ----------------------------

data = pd.DataFrame()
data["sleep_onset"]              = df[sleep_onset].map(freq_map)       if sleep_onset    else 0
data["sleep_duration"]           = df[sleep_duration].map(sleep_map)   if sleep_duration else 0
data["night_arousals"]           = df[night].map(freq_map)              if night          else 0
data["sleep_quality"]            = df[sleep_quality].map(quality_map)  if sleep_quality  else 0
data["difficulty_concentrating"] = df[concentration].map(freq_map)     if concentration  else 0
data["fatigue"]                  = df[fatigue].map(freq_map)            if fatigue        else 0
data["miss_classes"]             = df[miss_classes].map(miss_class_map) if miss_classes  else 0
data["miss_deadlines"]           = df[miss_deadlines].map(impact_map)  if miss_deadlines else 0
data["screen_time"]              = df[screen].map(freq_map)             if screen         else 0
data["caffeine"]                 = df[caffeine].map(freq_map)           if caffeine       else 0
data["exercise"]                 = df[exercise].map(freq_map)           if exercise       else 0
data["stress"]                   = df[stress].map(stress_map)           if stress         else 0

data["gpa"]    = df[gpa]    if gpa    else "Unknown"
data["gender"] = df[gender] if gender else "Unknown"

data = data.fillna(0)


# ----------------------------
# Normalise features for radar
# ----------------------------

features = [
    "sleep_onset", "sleep_duration", "night_arousals", "sleep_quality",
    "difficulty_concentrating", "fatigue", "miss_classes", "miss_deadlines",
    "screen_time", "caffeine", "exercise", "stress",
]

scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])


# ----------------------------
# Radar plot function
# ----------------------------

def radar_plot(group1, group2, labels, title, label1, label2, filename):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    g1 = np.concatenate([group1, [group1[0]]])
    g2 = np.concatenate([group2, [group2[0]]])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(angles, g1, label=label1)
    ax.plot(angles, g2, label=label2)
    ax.fill(angles, g1, alpha=0.1)
    ax.fill(angles, g2, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8)
    ax.set_title(title, pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    fig.tight_layout()
    fig.savefig(f"outputs/{dataset_type}/figures/{filename}", dpi=300)
    plt.close(fig)


os.makedirs(f"outputs/{dataset_type}/figures", exist_ok=True)


# ----------------------------
# Fig 6a: High vs Low GPA
# ----------------------------

high = data[data["gpa"].isin(["Good", "Excellent"])][features].mean().values
low  = data[data["gpa"].isin(["Poor", "Below Average", "Average"])][features].mean().values

radar_plot(
    high, low, features,
    "Behavioural Profile: High vs Low GPA",
    "High GPA", "Low GPA",
    "fig6a_high_vs_low_gpa.png",
)


# ----------------------------
# Fig 6b: Male vs Female
# ----------------------------

male   = data[data["gender"] == "Male"][features].mean().values
female = data[data["gender"] == "Female"][features].mean().values

radar_plot(
    male, female, features,
    "Behavioural Profile: Male vs Female",
    "Male", "Female",
    "fig6b_male_vs_female.png",
)

print("Fig 6a/6b: Radar plots saved.")
