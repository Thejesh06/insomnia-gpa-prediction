import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE


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
miss_classes   = get_col(["skip classes", "classes"])
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

deadline_map = {
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
data["miss_deadlines"]           = df[miss_deadlines].map(deadline_map) if miss_deadlines else 0
data["screen_time"]              = df[screen].map(freq_map)             if screen         else 0
data["caffeine"]                 = df[caffeine].map(freq_map)           if caffeine       else 0
data["exercise"]                 = df[exercise].map(freq_map)           if exercise       else 0
data["stress"]                   = df[stress].map(stress_map)           if stress         else 0

data["target"] = df[gpa].apply(lambda x: 1 if x in ["Good", "Excellent"] else 0)

data = data.apply(lambda col: col.fillna(col.median()))


# ----------------------------
# Features and target
# ----------------------------

X = data.drop(columns=["target"])
y = data["target"]


# ----------------------------
# Train / test split then scale
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


# ----------------------------
# SMOTE on training set only
# ----------------------------

if len(set(y_train)) > 1 and len(y_train) > 10:
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)


# ----------------------------
# Train SVM and compute permutation importance
# ----------------------------

model = SVC(kernel="rbf", probability=True, random_state=42)
model.fit(X_train, y_train)

perm = permutation_importance(
    model, X_test, y_test,
    n_repeats=30,
    random_state=42,
    scoring="roc_auc",
)

importance_df = pd.DataFrame({
    "Feature":    X.columns,
    "Importance": perm.importances_mean,
}).sort_values("Importance", ascending=False)

print("\nFig 5: Permutation Feature Importance — SVM (RBF)\n")
print(importance_df.to_string(index=False))


# ----------------------------
# Plot
# ----------------------------

plt.figure(figsize=(8, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="steelblue")
plt.xlabel("Mean Decrease in ROC AUC")
plt.title("Permutation Feature Importance — SVM (RBF)")
plt.gca().invert_yaxis()
plt.tight_layout()

os.makedirs(f"outputs/{dataset_type}/figures", exist_ok=True)
plt.savefig(f"outputs/{dataset_type}/figures/fig5_feature.jpg", dpi=300)
plt.close()
