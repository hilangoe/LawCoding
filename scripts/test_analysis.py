# test analysis script
import argparse
import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, cohen_kappa_score

# -----------------------------
# CLI arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Evaluate key detection and deontic agreement.")
parser.add_argument(
    "--version",
    required=True,
    help="Version tag for this analysis run (e.g. v1, stage1_pruned, instr_v2)"
)
args = parser.parse_args()

VERSION = args.version
print(f"Running analysis version: {VERSION}")

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"
merged_csv_path = BASE_DIR / "data" / "test_data_merged.csv"
merged_jsonl_path = BASE_DIR / "data" / "test_data_merged.jsonl"

# -----------------------------
# Load merged output
# -----------------------------
print("Loading merged predictions...")
if merged_csv_path.exists():
    df = pd.read_csv(merged_csv_path)
elif merged_jsonl_path.exists():
    df = pd.read_json(merged_jsonl_path, lines=True)
else:
    raise FileNotFoundError("Merged output not found!")

print(f"Total rows: {len(df)}")
print(df.head())

# -----------------------------
# Key frequency & agreement table
# -----------------------------

print("\nBuilding key frequency and agreement table...")

key_stats = (
    df
    .groupby("key")
    .agg(
        human_found=("human_key_present", "sum"),
        model_found=("model_key_present", "sum"),
        agreement_found=(
            "human_key_present",
            lambda x: (
                x & df.loc[x.index, "model_key_present"]
            ).sum()
        )
    )
    .reset_index()
)

key_stats["analysis_version"] = VERSION

# Derived error counts
key_stats["human_only"] = key_stats["human_found"] - key_stats["agreement_found"]
key_stats["model_only"] = key_stats["model_found"] - key_stats["agreement_found"]

# Precision & recall (safe divide)
key_stats["precision"] = key_stats.apply(
    lambda r: r["agreement_found"] / r["model_found"] if r["model_found"] > 0 else None,
    axis=1
)

key_stats["recall"] = key_stats.apply(
    lambda r: r["agreement_found"] / r["human_found"] if r["human_found"] > 0 else None,
    axis=1
)

# Sort by most problematic false positives
key_stats = key_stats.sort_values(
    by=["model_only", "model_found"],
    ascending=False
)

print("\nKey frequency & agreement (top 10 by false positives):")
print(key_stats.head(10))

# -----------------------------
# Save table
# -----------------------------
key_stats_path_csv = ANALYSIS_DIR / f"key_frequency_agreement_{VERSION}.csv"
key_stats_path_json = ANALYSIS_DIR / f"key_frequency_agreement_{VERSION}.json"

key_stats.to_csv(key_stats_path_csv, index=False)
key_stats.to_json(key_stats_path_json, orient="records", indent=2)

print(f"\nKey frequency table saved to:")
print(f"CSV  → {key_stats_path_csv}")
print(f"JSON → {key_stats_path_json}")

# -----------------------------
# Key detection metrics
# -----------------------------
y_true_key = df["human_key_present"]
y_pred_key = df["model_key_present"]

key_precision = precision_score(y_true_key, y_pred_key)
key_recall = recall_score(y_true_key, y_pred_key)
key_f1 = f1_score(y_true_key, y_pred_key)

print("\nKey detection metrics:")
print(f"Precision: {key_precision:.3f}")
print(f"Recall:    {key_recall:.3f}")
print(f"F1-score:  {key_f1:.3f}")

# -----------------------------
# Deontic agreement (only where key exists in both)
# -----------------------------
mask = df["human_key_present"] & df["model_key_present"]
df_deontic = df[mask]

y_true_deontic = df_deontic["human_deontic"]
y_pred_deontic = df_deontic["predicted_deontic"]

deontic_accuracy = (y_true_deontic == y_pred_deontic).mean()
conf_matrix = confusion_matrix(y_true_deontic, y_pred_deontic, labels=[-1, 1])
cohen_kappa = cohen_kappa_score(y_true_deontic, y_pred_deontic)

print("\nDeontic agreement (on keys both found):")
print(f"Accuracy: {deontic_accuracy:.3f}")
print("Confusion matrix (rows=true, cols=predicted):")
print(conf_matrix)
print(f"Cohen's kappa: {cohen_kappa:.3f}")

# breakdown by law or key
deontic_by_key = (
    df_deontic
    .assign(correct=lambda x: x["human_deontic"] == x["predicted_deontic"])
    .groupby("key")
    .agg(
        accuracy=("correct", "mean"),
        n=("correct", "size")
    )
    .sort_values("n", ascending=False)
)
print("\nDeontic agreement per key (accuracy):")
print(deontic_by_key)

# -----------------------------
# Save evaluation summary
# -----------------------------
summary = {
    "version": VERSION,
    "input_file": str(merged_csv_path if merged_csv_path.exists() else merged_jsonl_path),
    "key_precision": key_precision,
    "key_recall": key_recall,
    "key_f1": key_f1,
    "deontic_accuracy": deontic_accuracy,
    "cohen_kappa": cohen_kappa,
    "total_rows": len(df),
    "rows_deontic_eval": len(df_deontic),
}

summary_path = ANALYSIS_DIR / f"evaluation_summary_{VERSION}.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nEvaluation summary saved to {summary_path}")