import pandas as pd
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

def get_col(possible_names):
    for col in df.columns:
        for name in possible_names:
            if name.lower() in col.lower():
                return col
    return None


sleep_col = get_col(["sleep hours", "average"])
stress_col = get_col(["stress"])
gpa_col    = get_col(["gpa", "academic performance"])
year_col   = get_col(["year"])
dept_col   = get_col(["department", "dept"])


# ----------------------------
# Encoding maps
# ----------------------------

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

df["Sleep_Duration"] = df[sleep_col].map(sleep_map)   if sleep_col  else 0
df["Stress_Level"]   = df[stress_col].map(stress_map) if stress_col else 0
df["GPA_Score"]      = df[gpa_col].map(gpa_map)        if gpa_col    else 0
df["Year"]           = df[year_col]                    if year_col   else "Unknown"

df = df.fillna(0)

os.makedirs(f"outputs/{dataset_type}/tables", exist_ok=True)


# ----------------------------
# Table 3: Year-wise trends
# ----------------------------

table3 = df.groupby("Year")[["GPA_Score", "Sleep_Duration", "Stress_Level"]].mean().round(2)

print("\nTable 3: Academic and Behavioral Trends by Year\n")
print(table3.to_string())

table3.to_csv(f"outputs/{dataset_type}/tables/table3_result.csv")


# ----------------------------
# Table 3b: Department-wise trends
# ----------------------------

if dept_col:
    dept_table = df.groupby(dept_col)[["GPA_Score", "Sleep_Duration", "Stress_Level"]].mean().round(2)

    print("\nTable 3b: Trends by Department\n")
    print(dept_table.to_string())

    dept_table.to_csv(f"outputs/{dataset_type}/tables/table3b_department.csv")
else:
    print("\nDepartment column not found — Table 3b skipped.")
