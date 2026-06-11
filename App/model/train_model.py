import pandas as pd
import numpy as np
import pickle
import os

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DATA_PATH   = os.path.join(PROJECT_DIR, 'data', 'Student Insomnia and Educational Outcomes Dataset_version-2.csv')


# ── Load data ─────────────────────────────────────────────────────────────────

df = pd.read_csv(DATA_PATH)
print(f"Loaded survey data: {df.shape}")


# ── Column detection ──────────────────────────────────────────────────────────

def get_col(df, names):
    for col in df.columns:
        for n in names:
            if n.lower() in col.lower():
                return col
    return None


sleep_onset_col    = get_col(df, ['difficulty falling asleep'])
sleep_duration_col = get_col(df, ['sleep hours', 'average'])
night_col          = get_col(df, ['waking', 'wake'])
screen_col         = get_col(df, ['screen', 'electronic'])
caffeine_col       = get_col(df, ['caffeine'])
exercise_col       = get_col(df, ['exercise', 'physical activity'])
stress_col         = get_col(df, ['stress'])
gpa_col            = get_col(df, ['gpa', 'academic performance'])


# ── Encoding maps ─────────────────────────────────────────────────────────────

freq_map = {
    'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4, 'Always': 5,
    'Every day': 5, 'Every night': 5,
    'Rarely (1-2 times a week)': 2,
    'Sometimes (3-4 times a week)': 3,
    'Often (5-6 times a week)': 4,
}

sleep_map = {
    'Less than 4 hours': 1, '4-5 hours': 2, '5-6 hours': 2,
    '6-7 hours': 3, '7-8 hours': 4, 'More than 8 hours': 5,
}

stress_map = {
    'No stress': 1, 'Low stress': 2, 'High stress': 3, 'Extremely high stress': 4,
}


# ── Build feature matrix ──────────────────────────────────────────────────────

data = pd.DataFrame()
data['sleep_onset']    = df[sleep_onset_col].map(freq_map)
data['sleep_duration'] = df[sleep_duration_col].map(sleep_map)
data['night_arousals'] = df[night_col].map(freq_map)
data['screen_time']    = df[screen_col].map(freq_map)
data['caffeine']       = df[caffeine_col].map(freq_map)
data['exercise']       = df[exercise_col].map(freq_map)
data['stress']         = df[stress_col].map(stress_map)
data['target']         = df[gpa_col].apply(lambda x: 1 if x in ['Good', 'Excellent'] else 0)

data = data.apply(lambda col: col.fillna(col.median()))

FEATURES = ['sleep_onset', 'sleep_duration', 'night_arousals',
            'screen_time', 'caffeine', 'exercise', 'stress']

X = data[FEATURES].values
y = data['target'].values

print(f"Class distribution (original): {dict(zip(*np.unique(y, return_counts=True)))}")


# ── Scale → SMOTE → Train ─────────────────────────────────────────────────────

scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_scaled, y)
print(f"Class distribution (after SMOTE): {dict(zip(*np.unique(y_smote, return_counts=True)))}")

model = RandomForestClassifier(random_state=42)
model.fit(X_smote, y_smote)
print("Model trained.")


# ── Save artifacts ────────────────────────────────────────────────────────────

with open(os.path.join(SCRIPT_DIR, 'smote_model.pkl'), 'wb') as f:
    pickle.dump(model, f)

with open(os.path.join(SCRIPT_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

# Class-average profiles (unscaled) used by the app's radar chart
profiles = {
    'features':       FEATURES,
    'high_gpa_mean':  X[y == 1].mean(axis=0).tolist(),
    'low_gpa_mean':   X[y == 0].mean(axis=0).tolist(),
}
with open(os.path.join(SCRIPT_DIR, 'feature_profiles.pkl'), 'wb') as f:
    pickle.dump(profiles, f)

print(f"Saved: App/model/smote_model.pkl")
print(f"Saved: App/model/scaler.pkl")
print(f"Saved: App/model/feature_profiles.pkl")
print(f"Features: {FEATURES}")
