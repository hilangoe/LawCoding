# this script will execute the pipeline to prep the training data
from training_prep_library, import process_multiple_laws, create_training_rows
from synthetic_data, import generate_all_synthetic_rows


# loading the codebook JSON once
with open("data2/cb_expression_rights.json", "r") as f:
    codebook = json.load(f)

# define law_list

# define model

# define pdf_folder path

# define human_folder

# processing the training set
raw_results = process_multiple_laws(law_list, model, pdf_folder, human_folder, codebook)

# prepping output for fine-tuning

# defining the labels here just in case
DEONTIC_LABELS = [-1, 1]
rows, key_to_id, deontic_to_id = create_training_rows(raw_results, codebook, deontic_labels = DEONTIC_LABELS)

# save training rows to disk as JSONL
output_file = "training_data.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for row in rows:
        json_line = json.dumps(row, ensure_ascii=False)
        f.write(json_line + "\n")

print(f"Saved {len(rows)} rows to {output_file}")

# creating metadata json
build_minimal_metadata(codebook, "metadata.json")