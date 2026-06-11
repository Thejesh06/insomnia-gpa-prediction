import pandas as pd
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
night          = get_col(["waking", "wake"])
sleep_duration = get_col(["sleep hours", "average"])
gender         = get_col(["gender"])
year           = get_col(["year"])
dept           = get_col(["department", "dept"])


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


# ----------------------------
# Apply encoding
# ----------------------------

df["sleep_onset"]    = df[sleep_onset].map(freq_map)    if sleep_onset    else 0
df["night_arousals"] = df[night].map(freq_map)           if night          else 0
df["sleep_duration"] = df[sleep_duration].map(sleep_map) if sleep_duration else 0

df[["sleep_onset", "night_arousals", "sleep_duration"]] = (
    df[["sleep_onset", "night_arousals", "sleep_duration"]].fillna(0)
)


# ----------------------------
# DSM-inspired insomnia flag
# ----------------------------

df["insomnia"] = (
    (df["sleep_onset"] >= 3) &
    (df["night_arousals"] >= 3) &
    (df["sleep_duration"] < 6)
).astype(int)

os.makedirs(f"outputs/{dataset_type}/figures", exist_ok=True)


# ----------------------------
# Fig 1a: Insomnia by gender
# ----------------------------

if gender:
    gender_counts = pd.crosstab(df[gender], df["insomnia"])
    gender_counts.plot(kind="bar", figsize=(6, 4))
    plt.title("Insomnia by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Number of Students")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"outputs/{dataset_type}/figures/fig1a_insomnia.jpg", dpi=300)
    plt.close()
else:
    print("Gender column not found — Fig 1a skipped.")


# ----------------------------
# Fig 1b: Insomnia by academic year
# ----------------------------

if year:
    year_counts = pd.crosstab(df[year], df["insomnia"])
    year_counts.plot(kind="bar", figsize=(6, 4))
    plt.title("Insomnia by Academic Year")
    plt.xlabel("Academic Year")
    plt.ylabel("Number of Students")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"outputs/{dataset_type}/figures/fig1b_result.jpg", dpi=300)
    plt.close()
else:
    print("Year column not found — Fig 1b skipped.")


# ----------------------------
# Fig 1c: Insomnia by department
# ----------------------------

if dept:
    dept_counts = pd.crosstab(df[dept], df["insomnia"])
    dept_counts.plot(kind="bar", figsize=(7, 4))
    plt.title("Insomnia by Department")
    plt.xlabel("Department")
    plt.ylabel("Number of Students")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"outputs/{dataset_type}/figures/fig1c_department.jpg", dpi=300)
    plt.close()
else:
    print("Department column not found — Fig 1c skipped.")

print("Fig 1a/1b/1c: Insomnia distribution plots saved.")
