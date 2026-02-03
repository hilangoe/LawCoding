# this is the script for generating graphs for the training data
import json
import requests
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "train.csv"
JSON_DIR = BASE_DIR / "data" / "laws_json"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# Load training CSV and get law list
# -----------------------------
df_train = pd.read_csv(CSV_PATH)
law_list = df_train["path"].apply(lambda p: Path(p).stem).dropna().unique().tolist()

# -----------------------------
# Load codebook from Hugging Face
# -----------------------------
CODEBOOK_URL = "https://huggingface.co/spaces/raulpzs/expression_laws/resolve/main/data3/codebook.json"
CODEBOOK_FILE = BASE_DIR / "cb_expression_rights.json"

# download if not already present
if not CODEBOOK_FILE.exists():
    response = requests.get(CODEBOOK_URL)
    response.raise_for_status()
    with open(CODEBOOK_FILE, "wb") as f:
        f.write(response.content)

with open(CODEBOOK_FILE, "r", encoding="utf-8") as f:
    codebook = json.load(f)

# extract all provision keys (master list)
keys = set()
for entry in codebook:
    for actor in (entry.get("Actors") or {}).values():
        if actor.get("Key"):
            keys.add(actor["Key"])
keys = sorted(keys)
print(f"Total provision keys from codebook: {len(keys)}")

# -----------------------------
# Initialize containers
# -----------------------------
provision_counts = defaultdict(int)  # provision_name -> number of laws
law_counts = []                       # number of real provisions per law

# -----------------------------
# Loop through laws in training list
# -----------------------------
for law_id in law_list:
    json_file = JSON_DIR / f"{law_id}.json"
    if not json_file.exists():
        continue

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    law_real_count = 0

    for provision in data.get("Provisions", []):
        prov_name = provision.get("Provision")
        code = provision.get("Code")

        if prov_name and code in [-1, 1]:
            provision_counts[prov_name] += 1
            law_real_count += 1

    law_counts.append(law_real_count)

# -----------------------------
# Ensure all provision keys are represented
# -----------------------------
prov_series = pd.Series(provision_counts)
prov_series = prov_series.reindex(keys, fill_value=0)

# -----------------------------
# Plot 1: per-provision distribution
# -----------------------------

prov_df = prov_series.sort_values(ascending=False).reset_index()
prov_df.columns = ["Provision", "n_laws"]

# printing number of provisions below threshold of 10
num_provisions_below_10 = (prov_df['n_laws'] < 10).sum()
print(f"Number of provisions with fewer than 10 samples: {num_provisions_below_10}")

# printing top 20 and bottom 20 provisions
top20 = prov_df.head(20)
bottom20 = prov_df.tail(20)

print("Top 20 provision keys (most laws with real samples):")
print(top20)

print("\nBottom 20 provision keys (fewest laws with real samples):")
print(bottom20)

plt.figure(figsize=(8, 4))  # narrower width
plt.bar(range(len(prov_df)), prov_df["n_laws"], width=0.9)
plt.xlabel("Provision (sorted by frequency)")
plt.ylabel("Number of laws with real data")
plt.title("Real sample count per provision")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "provision_distribution_training.png", dpi=300)
plt.close()
print(f"Per-provision plot saved to {OUTPUT_DIR / 'provision_distribution_training.png'}")

# -----------------------------
# Plot 2: per-law distribution
# -----------------------------
plt.figure(figsize=(8,5))
sns.histplot(
    law_counts,
    bins=range(0, max(law_counts)+2),  # one bin per integer
    color="salmon",
    edgecolor="black"
)
plt.xlabel("Number of provisions")
plt.ylabel("Number of laws")
plt.title("Distribution of real samples per law (training set)")
# plt.xticks(range(0, max(law_counts)+1))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "law_distribution_training.png", dpi=300)
plt.close()
print(f"Per-law plot saved to {OUTPUT_DIR / 'law_distribution_training.png'}")
