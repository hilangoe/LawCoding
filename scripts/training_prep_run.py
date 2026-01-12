# this script will execute the pipeline to prep the training data
from pathlib import Path
import pandas as pd
import json, requests

# importing local modules
from training_prep_library import process_multiple_laws, create_training_rows, build_metadata
from synthetic_data import generate_all_synthetic_rows, sparse_keys

# Grabbing the codebook from the huggingface repo
codebook_url = "https://huggingface.co/spaces/raulpzs/expression_laws/resolve/main/data3/codebook.json"
filename = "cb_expression_rights.json"

response = requests.get(codebook_url)
response.raise_for_status()

# downloading the codebook JSON
with open(filename, "wb") as f:
    f.write(response.content)

# loading the codebook JSON once
with open("cb_expression_rights.json", "r") as f:
    codebook = json.load(f)

# define model
model = "gpt-5"

# setting path for train list
csv_path = "../data/train.csv"   # adjust if needed

df = pd.read_csv(csv_path)

# creating law_list
law_list = (
    df["path"]
      .apply(lambda p: Path(p).stem)  # drop folder + .json
      .dropna()
      .unique()
      .tolist()
)

# setting base path
BASE_DIR = Path(__file__).resolve().parent.parent # going back up one level to the project folder

# defining pdf and json folder paths
pdf_dir = BASE_DIR / "data" / "laws_pdf"
json_dir = BASE_DIR / "data" / "laws_json"

## processing the training set
raw_results = process_multiple_laws(
    law_list=law_list,
    model=model,
    pdf_dir=pdf_dir,
    json_dir=json_dir,
    codebook=codebook
)

## prepping output for fine-tuning
# defining the labels here just in case
DEONTIC_LABELS = [-1, 1]

# setting threshold for minimal samples needed per key
threshold = 10

# calculating needed synthetic samples per key
key_counts = sparse_keys(raw_results["Successes"],
                         threshold=threshold)

if key_counts:
    print("Keys needing synthetic data (current count < threshold):")
    print(f"{'Key':<25} {'Current Count':<15} {'Needed':<10}")
    print("-" * 55)
    for key, count in key_counts.items():
        needed = threshold - count  # how many synthetic samples to generate
        print(f"{key:<25} {count:<15} {needed:<10}")
else:
    print("All keys meet or exceed the threshold. No synthetic data needed.")

# creating synthetic data
synth_rows = generate_all_synthetic_rows(
    key_counts=key_counts,
    codebook=codebook,
    model=model,
    threshold=threshold
)

all_raw_rows = raw_results["Successes"] + synth_rows


rows_final, key_to_id, deontic_to_id = create_training_rows(law_outputs=all_raw_rows, 
                                                            codebook=codebook, 
                                                            deontic_labels = DEONTIC_LABELS)


# save training rows to disk as JSONL
output_dir = BASE_DIR / "data"
output_dir.mkdir(exist_ok=True)

output_file = "training_data.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for row in rows_final:
        json_line = json.dumps(row, ensure_ascii=False)
        f.write(json_line + "\n")

print(f"Saved {len(rows_final)} rows to {output_file}")

# creating metadata json
meta_data_file = "../data/metadata.json"

build_metadata(codebook, meta_data_file)