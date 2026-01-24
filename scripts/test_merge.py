# -----------------------------
# Stage 3: Merge Stage 2 predictions with human-coded JSONs
# -----------------------------

import json
from pathlib import Path
from collections import Counter
import pandas as pd

# -----------------------------
# Base directory & data paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # project folder
data_path = BASE_DIR / "data"
data_path.mkdir(parents=True, exist_ok=True)

stage2_path = data_path / "test_predictions.jsonl"  # Stage 2 predictions
human_json_dir = data_path / "law_json"             # folder of human-coded JSONs
output_file = data_path / "merged_law_predictions.json"

# -----------------------------
# Reconstruct test set
# -----------------------------
csv_path = data_path / "test.csv"
df = pd.read_csv(csv_path)

law_list = (
    df["path"]
      .apply(lambda p: Path(p).stem)  # drop folder + .json
      .dropna()
      .unique()
      .tolist()
)

print(f"Test set contains {len(law_list)} laws.")

# -----------------------------
# Load human-coded JSONs
# -----------------------------
human_data_by_law = {}
for json_file in human_json_dir.glob("*.json"):
    law_id = json_file.stem
    if law_id in law_list:  # only keep laws in the test set
        with open(json_file, "r", encoding="utf-8") as f:
            human_data_by_law[law_id] = json.load(f)

print(f"Loaded {len(human_data_by_law)} human-coded laws.")

# -----------------------------
# Load Stage 2 predictions
# -----------------------------
stage2_predictions = []
with open(stage2_path, "r", encoding="utf-8") as f:
    for line in f:
        stage2_predictions.append(json.loads(line))

# Group predictions by law_id
preds_by_law = {}
for pred in stage2_predictions:
    law_id = pred["law_id"]
    if law_id in law_list:
        preds_by_law.setdefault(law_id, []).append(pred)

# -----------------------------
# Load codebook and build label maps
# -----------------------------
codebook_path = data_path / "codebook.json"
with open(codebook_path, "r", encoding="utf-8") as f:
    codebook = json.load(f)

def get_label_maps(codebook, deontic_labels=None):
    ontology_keys = set()
    for entry in codebook:
        actors = entry.get("Actors", {}) or {}
        for actor_info in actors.values():
            key = actor_info.get("Key")
            if key:
                ontology_keys.add(key)

    sorted_keys = sorted(ontology_keys)
    key_to_id = {k: i for i, k in enumerate(sorted_keys)}
    id_to_key = {str(i): k for i, k in enumerate(sorted_keys)}

    if deontic_labels is None:
        deontic_labels = [-1, 1]

    deontic_to_id = {d: i for i, d in enumerate(deontic_labels)}
    id_to_deontic = {str(i): d for i, d in enumerate(deontic_labels)}

    return {
        "key": {"label_to_id": key_to_id, "id_to_label": id_to_key},
        "deontic": {"label_to_id": deontic_to_id, "id_to_label": id_to_deontic},
    }

label_maps = get_label_maps(codebook)
deontic_id_to_label = label_maps["deontic"]["id_to_label"]

# -----------------------------
# Merge predictions with human data
# -----------------------------
merged_laws = []

for law_id, human_law in human_data_by_law.items():
    law_preds = preds_by_law.get(law_id, [])
    preds_by_key = {}
    for p in law_preds:
        key = p.get("predicted_key")
        if key is not None:
            preds_by_key.setdefault(key, []).append(p)

    aggregated_provisions = []
    for prov in human_law["Provisions"]:
        key = prov["Provision"]
        preds_for_key = preds_by_key.get(key, [])

        # concatenate texts
        predicted_text = "\n\n".join(p.get("text", "").strip() for p in preds_for_key) or None

        # majority vote for deontic
        deontics = [p.get("predicted_deontic") for p in preds_for_key if p.get("predicted_deontic") is not None]
        predicted_deontic = deontic_id_to_label[str(Counter(deontics).most_common(1)[0][0])] if deontics else None

        aggregated_provisions.append({
            "Provision": key,
            "human_text": prov.get("Note"),
            "human_deontic": prov.get("Code"),
            "predicted_text": predicted_text,
            "predicted_deontic": predicted_deontic,
            "predicted_key_present": bool(preds_for_key)
        })

    merged_laws.append({
        "law_id": law_id,
        "File": human_law.get("File"),
        "Law": human_law.get("Law"),
        "Year": human_law.get("Year"),
        "Language": human_law.get("Language"),
        "Country": human_law.get("Country"),
        "Provisions": aggregated_provisions
    })

# -----------------------------
# Save merged JSON
# -----------------------------
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(merged_laws, f, ensure_ascii=False, indent=2)

print(f"Merged laws saved to {output_file}. Total laws: {len(merged_laws)}")

