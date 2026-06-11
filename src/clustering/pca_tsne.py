import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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
stress         = get_col(["stress"])
caffeine       = get_col(["caffeine"])
screen         = get_col(["screen", "electronic"])
exercise       = get_col(["exercise", "physical activity"])


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
    "No stress": 1, "Low stress": 2, "High stress": 3, "Extremely high stress": 4,
}


# ----------------------------
# Build feature dataframe
# ----------------------------

data = pd.DataFrame()
data["sleep_onset"]    = df[sleep_onset].map(freq_map)    if sleep_onset    else 0
data["sleep_duration"] = df[sleep_duration].map(sleep_map) if sleep_duration else 0
data["night_arousals"] = df[night].map(freq_map)           if night          else 0
data["stress"]         = df[stress].map(stress_map)        if stress         else 0
data["caffeine"]       = df[caffeine].map(freq_map)        if caffeine       else 0
data["screen_time"]    = df[screen].map(freq_map)          if screen         else 0
data["exercise"]       = df[exercise].map(freq_map)        if exercise       else 0

data = data.apply(lambda col: col.fillna(col.median()))


# ----------------------------
# Standardise and cluster
# ----------------------------

features = list(data.columns)
X_scaled = StandardScaler().fit_transform(data[features])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

os.makedirs(f"outputs/{dataset_type}/figures", exist_ok=True)


# ----------------------------
# Fig 4a: PCA projection
# ----------------------------

X_pca = PCA(n_components=2).fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(7, 6))
for c in range(3):
    idx = clusters == c
    ax.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f"Cluster {c}", alpha=0.6)

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_title("Fig. 4a — PCA Projection of Behavioural Clusters")
ax.legend()
fig.tight_layout()
fig.savefig(f"outputs/{dataset_type}/figures/fig4a_pca.jpg", dpi=300)
plt.close(fig)


# ----------------------------
# Fig 4b: t-SNE projection
# ----------------------------

X_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(7, 6))
for c in range(3):
    idx = clusters == c
    ax.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=f"Cluster {c}", alpha=0.6)

ax.set_xlabel("t-SNE Dimension 1")
ax.set_ylabel("t-SNE Dimension 2")
ax.set_title("Fig. 4b — t-SNE Projection of Behavioural Clusters")
ax.legend()
fig.tight_layout()
fig.savefig(f"outputs/{dataset_type}/figures/fig4b_t-sne.jpg", dpi=300)
plt.close(fig)

print("Fig 4a/4b: PCA and t-SNE projection plots saved.")
