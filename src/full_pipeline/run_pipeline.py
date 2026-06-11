import sys
import os
import subprocess

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(BASE_DIR)
os.chdir(BASE_DIR)

from src.preprocessing.data_preprocessing import load_and_preprocess


# ----------------------------
# Dataset selection
# ----------------------------

if len(sys.argv) < 2:
    print("Usage: python run_pipeline.py [base|survey]")
    sys.exit(1)

dataset_type = sys.argv[1]

if dataset_type == "base":
    data_path = os.path.join(BASE_DIR, "data", "Student Insomnia and Educational Outcomes Dataset_version-2.csv")
elif dataset_type == "survey":
    data_path = os.path.join(BASE_DIR, "data", "survey_dataset.csv")
else:
    print("Invalid dataset type. Use 'base' or 'survey'.")
    sys.exit(1)

print(f"\nRunning pipeline for: {dataset_type.upper()} dataset\n")


# ----------------------------
# Step 1: Preprocess and save
# ----------------------------

df = load_and_preprocess(data_path)
print(f"Data loaded — shape: {df.shape}")

os.makedirs("outputs/processed", exist_ok=True)
df.to_csv(f"outputs/processed/{dataset_type}_clean.csv", index=False)
print(f"Saved: outputs/processed/{dataset_type}_clean.csv\n")


# ----------------------------
# Script runner
# ----------------------------

def run_script(script_path):
    print(f"Running: {script_path}")
    result = subprocess.run([sys.executable, script_path, data_path, dataset_type])
    if result.returncode != 0:
        print(f"Error: {script_path} exited with code {result.returncode}")
        sys.exit(1)


# ----------------------------
# Step 2: Analysis
# ----------------------------

run_script("src/analysis/demographic_analysis.py")
run_script("src/analysis/behavioral_analysis.py")
run_script("src/analysis/table4_grpgpa.py")
run_script("src/analysis/year_trends_table3.py")


# ----------------------------
# Step 3: Visualisation
# ----------------------------

run_script("src/visualization/fig2_correlation_heatmap.py")
run_script("src/visualization/radar_plots.py")
run_script("src/visualization/feature_imp.py")
run_script("src/visualization/insomnia_analysis.py")


# ----------------------------
# Step 4: Models
# ----------------------------

if dataset_type == "base":
    run_script("src/models/table5_LR.py")   # base dataset only — uses 13-feature LR replication

run_script("src/models/table6_unbalanced_models.py")
run_script("src/models/table7_smote.py")
run_script("src/models/table8_smote_corrected.py")


# ----------------------------
# Step 5: Clustering
# ----------------------------

run_script("src/clustering/kmeans_clustering.py")
run_script("src/clustering/pca_tsne.py")


# ----------------------------
# Done
# ----------------------------

print(f"\nPipeline complete — {dataset_type.upper()} dataset processed successfully.")
