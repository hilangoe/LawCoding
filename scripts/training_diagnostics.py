# this script is for running diagnostics on generated training data, comparing it to the raw JSON files
import json, os

import pandas as pd

from pathlib import Path

from extract_text import _ensure_text

# pull in training list
# setting path for train list
csv_path = "../data/train.csv"   # adjust if needed

df = pd.read_csv(csv_path)
#print(df.shape)

# creating law_list for handling the files
law_list = (
    df["path"]
      .apply(lambda p: Path(p).stem)  # drop folder + .json
      .dropna()
      .unique()
      .tolist()
)

#print(len(law_list))

# setting up dataframe for importing features
df["law_id"] = df["path"].apply(lambda p: Path(p).stem)
df = df.drop("path", axis=1)

# setting up file path
BASE_DIR = Path(__file__).resolve().parent.parent # going back up one level to the project folder
json_dir = BASE_DIR / "data" / "laws_json"
pdf_dir = BASE_DIR / "data" / "laws_pdf"

# initializing list of dictionaries
law_data = []

# 1: create text length
# 2: number of provisions in JSON
for law_id in law_list:
    # Look for PDF
    all_files = os.listdir(pdf_dir)
    pdf_candidates = [
        os.path.join(pdf_dir, f)
        for f in all_files
        if law_id.lower() in f.lower() and f.lower().endswith(".pdf")
    ]
    
    pdf_path = None
    if pdf_candidates:
        # prefer English PDF if available
        pdf_path = next((p for p in pdf_candidates if "ENG" in os.path.basename(p).upper()), pdf_candidates[0])
    
    if pdf_path:
        law_text = _ensure_text(pdf_path)
        pdf_missing = False
    else:
        law_text = ""
        pdf_missing = True

    law_length = len(law_text) if law_text else 0

    # JSON
    json_path = os.path.join(json_dir, f"{law_id}.json")
    key_count = 0
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        provisions = data.get("Provisions", [])
        for p in provisions:
            deontic = p.get("Code", None)
            if deontic in [-1, 1]:
                key_count += 1
    else:
        provisions = []
        key_count = 0

    law_data.append({
        "law_id": law_id,
        "law_length": law_length,
        "key_count_json": key_count,
        "pdf_missing": pdf_missing
    })

df_json_data = pd.DataFrame(law_data)

# 3: number of provisions in training data
data_path = Path("../data/training_data.jsonl")

rows = []
with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

df_rows = pd.DataFrame(rows)

df_training_data = df_rows.groupby("law_id").size().reset_index(name="key_count_training")

df_merged = pd.merge(df, df_json_data, on="law_id", how="left")
df_merged = pd.merge(df_merged, df_training_data, on="law_id", how="left")

# cleaning up
df_merged["key_count_training"] = (
    df_merged["key_count_training"]
    .fillna(0)
    .astype(int)
)

# adding loss column so it's easy to see how much we're losing per law
df_merged["loss"] = (
    df_merged["key_count_json"] - df_merged["key_count_training"]
)

# diagnostic: mean loss by PDF availability
mean_loss_by_pdf = (
    df_merged
    .groupby("pdf_missing")["loss"]
    .mean()
)

print("\nMean loss by PDF availability:")
for pdf_missing, mean_loss in mean_loss_by_pdf.items():
    status = "PDF missing" if pdf_missing else "PDF present"
    print(f"{status:<15}: {mean_loss:.2f}")

output_file = "training_diagnostics.csv"

df_merged.to_csv(output_file)