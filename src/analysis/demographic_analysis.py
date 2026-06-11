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


gender_col = get_col(["gender"])
year_col   = get_col(["year"])
dept_col   = get_col(["department", "dept"])

os.makedirs(f"outputs/{dataset_type}/tables", exist_ok=True)

print("\nTable 1: Demographic Distribution\n")


# ----------------------------
# Gender distribution
# ----------------------------

if gender_col:
    gender_table = pd.DataFrame({
        "Count":          df[gender_col].value_counts(),
        "Percentage (%)": df[gender_col].value_counts(normalize=True).mul(100).round(2),
    })
    print("Gender Distribution:\n")
    print(gender_table.to_string())
    gender_table.to_csv(f"outputs/{dataset_type}/tables/table1a_result.csv")
else:
    print("Gender column not found — skipped.")

print()


# ----------------------------
# Year distribution
# ----------------------------

if year_col:
    year_table = pd.DataFrame({
        "Count":          df[year_col].value_counts(),
        "Percentage (%)": df[year_col].value_counts(normalize=True).mul(100).round(2),
    })
    print("Academic Year Distribution:\n")
    print(year_table.to_string())
    year_table.to_csv(f"outputs/{dataset_type}/tables/table1b_result.csv")
else:
    print("Year column not found — skipped.")

print()


# ----------------------------
# Department distribution
# ----------------------------

if dept_col:
    dept_table = pd.DataFrame({
        "Count":          df[dept_col].value_counts(),
        "Percentage (%)": df[dept_col].value_counts(normalize=True).mul(100).round(2),
    })
    print("Department Distribution:\n")
    print(dept_table.to_string())
    dept_table.to_csv(f"outputs/{dataset_type}/tables/table1c_result.csv")
else:
    print("Department column not found — skipped.")
