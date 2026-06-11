import pandas as pd
import numpy as np
import sys
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, auc,
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


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
    print(f"Warning: column not found for {names}")
    return None


sleep_onset   = get_col(["difficulty falling asleep"])
sleep_duration = get_col(["sleep hours", "average"])
night         = get_col(["waking", "wake"])
screen        = get_col(["screen", "electronic"])
caffeine      = get_col(["caffeine"])
exercise      = get_col(["exercise", "physical activity"])
stress        = get_col(["stress"])
gpa           = get_col(["gpa", "academic performance"])


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


# ----------------------------
# Build feature dataframe
# ----------------------------

data = pd.DataFrame()
data["sleep_onset"]    = df[sleep_onset].map(freq_map)   if sleep_onset    else 0
data["sleep_duration"] = df[sleep_duration].map(sleep_map) if sleep_duration else 0
data["night_arousals"] = df[night].map(freq_map)          if night          else 0
data["screen_time"]    = df[screen].map(freq_map)         if screen         else 0
data["caffeine"]       = df[caffeine].map(freq_map)       if caffeine       else 0
data["exercise"]       = df[exercise].map(freq_map)       if exercise       else 0
data["stress"]         = df[stress].map(stress_map)       if stress         else 0

data["target"] = df[gpa].apply(lambda x: 1 if x in ["Good", "Excellent"] else 0)

data = data.apply(lambda col: col.fillna(col.median()))


# ----------------------------
# Features and target
# ----------------------------

X = data.drop(columns=["target"])
y = data["target"]


# ----------------------------
# Standardise
# ----------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ----------------------------
# SMOTE oversampling
# ----------------------------

if len(set(y)) > 1 and len(y) > 10:
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_scaled, y)
else:
    X_smote, y_smote = X_scaled, y.values


# ----------------------------
# Models
# ----------------------------

models = {
    "Decision Tree":      DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (RBF)":          SVC(kernel="rbf", probability=True),
    "k-NN":               KNeighborsClassifier(),
    "Random Forest":      RandomForestClassifier(random_state=42),
    "Naive Bayes":        GaussianNB(),
}


# ----------------------------
# Stratified cross-validation
# ----------------------------

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
results = []

plt.figure()

for name, model in models.items():

    acc_scores, prec_scores, rec_scores = [], [], []
    spec_scores, f1_scores, auc_scores  = [], [], []

    for train_idx, test_idx in skf.split(X_smote, y_smote):

        X_train, X_test = X_smote[train_idx], X_smote[test_idx]
        y_train, y_test = y_smote[train_idx], y_smote[test_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc_scores.append(accuracy_score(y_test, y_pred))
        prec_scores.append(precision_score(y_test, y_pred, zero_division=0))
        rec_scores.append(recall_score(y_test, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        spec_scores.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

        auc_scores.append(roc_auc_score(y_test, y_prob))

    # Refit on full SMOTE data for ROC visualisation
    model.fit(X_smote, y_smote)
    y_prob_full = model.predict_proba(X_smote)[:, 1]
    fpr, tpr, _ = roc_curve(y_smote, y_prob_full)
    roc_auc_val = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_val:.2f})")

    results.append([
        name,
        np.mean(acc_scores),
        np.mean(prec_scores),
        np.mean(rec_scores),
        np.mean(spec_scores),
        np.mean(f1_scores),
        np.mean(auc_scores),
    ])


# ----------------------------
# ROC curve plot
# ----------------------------

plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison — SMOTE Balanced Models")
plt.legend()
plt.tight_layout()

os.makedirs(f"outputs/{dataset_type}/figures", exist_ok=True)
plt.savefig(f"outputs/{dataset_type}/figures/roc_all_models.jpg", dpi=300)
plt.close()


# ----------------------------
# Results table
# ----------------------------

results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "Specificity", "F1 Score", "AUC"
]).round(3)

print("\nTable 7: Classifier Performance (SMOTE Balanced)\n")
print(results_df.to_string(index=False))

os.makedirs(f"outputs/{dataset_type}/tables", exist_ok=True)
results_df.to_csv(f"outputs/{dataset_type}/tables/table7_result.csv", index=False)
