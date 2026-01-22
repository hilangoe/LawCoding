# this is the final script for the pipeline test, taking inference output and merging in JSON data with actual labels
import json
from pathlib import Path
from collections import Counter

# -----------------------------
# Label mapping utility
# -----------------------------
def get_label_maps(codebook, deontic_labels=None, save_path=None):
    """
    Build key and deontic label mappings from a codebook.

    Args:
        codebook (list[dict]): Codebook containing ontology entries with "Actors" and "Key".
        deontic_labels (list, optional): List of deontic categories. Defaults to [-1, 1].
        save_path (str, optional): File path to save JSON output. If None, no file is written.

    Returns:
        dict: Structured dictionary with label_to_id and id_to_label for keys and deontics.
    """
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

    label_maps = {
        "key": {
            "label_to_id": key_to_id,
            "id_to_label": id_to_key
        },
        "deontic": {
            "label_to_id": deontic_to_id,
            "id_to_label": id_to_deontic
        }
    }

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(label_maps, f, ensure_ascii=False, indent=2)

    return label_maps

# -----------------------------
# Paths
# -----------------------------
stage2_path = Path("data/stage2_predictions.jsonl")  # JSONL from Stage 2
human_json_dir = Path("data/human_jsons")            # folder of human-coded JSONs
codebook_file = Path("data/codebook.json")           # the codebook for label mapping
output_file = Path("data/merged_law_predictions.json")

# -----------------------------
# Load codebook and build label maps
# -----------------------------
with open(codebook_file, "r", encoding="utf-8") as f:
    codebook = json.load(f)

label_maps = get_label_maps(codebook)
deontic_id_to_label = label_maps["deontic"]["id_to_label"]

# -----------------------------
# Load Stage 2 predictions
# -----------------------------
print("Loading Stage 2 predictions...")
stage2_predictions = []
with open(stage2_path, "r", encoding="utf-8") as f:
    for line in f:
        stage2_predictions.append(json.loads(line))

preds_by_law = {}
for pred in stage2_predictions:
    law_id = pred["law_id"]
    preds_by_law.setdefault(law_id, []).append(pred)

# -----------------------------
# Load human-coded JSONs
# -----------------------------
print("Loading human-coded JSONs...")
human_json_files = list(human_json_dir.glob("*.json"))
human_data_by_law = {}
for file_path in human_json_files:
    with open(file_path, "r", encoding="utf-8") as f:
        law_obj = json.load(f)
        law_id = Path(law_obj["File"]).stem
        human_data_by_law[law_id] = law_obj

# -----------------------------
# Aggregate per law and per key
# -----------------------------
merged_laws = []

for law_id, human_law in human_data_by_law.items():
    law_preds = preds_by_law.get(law_id, [])
    preds_by_key = {}
    for p in law_preds:
        key = p.get("predicted_key")
        if not key:
            continue
        preds_by_key.setdefault(key, []).append(p)

    aggregated_provisions = []
    for prov in human_law["Provisions"]:
        key = prov["Provision"]
        preds_for_key = preds_by_key.get(key, [])

        # concatenate texts
        predicted_text = "\n\n".join(p.get("text", "").strip() for p in preds_for_key)

        # majority vote for deontic
        deontics = [p.get("predicted_deontic") for p in preds_for_key if p.get("predicted_deontic") is not None]
        if deontics:
            majority_vote = Counter(deontics).most_common(1)[0][0]
            predicted_deontic = deontic_id_to_label[str(majority_vote)]
            predicted_key_present = True
        else:
            predicted_deontic = None
            predicted_key_present = False

        aggregated_provisions.append({
            "Provision": key,
            "human_text": prov.get("Note"),
            "human_deontic": prov.get("Code"),
            "predicted_text": predicted_text if predicted_text else None,
            "predicted_deontic": predicted_deontic,
            "predicted_key_present": predicted_key_present
        })

    merged_law = {
        "law_id": law_id,
        "File": human_law.get("File"),
        "Law": human_law.get("Law"),
        "Year": human_law.get("Year"),
        "Language": human_law.get("Language"),
        "Country": human_law.get("Country"),
        "Provisions": aggregated_provisions
    }

    merged_laws.append(merged_law)

# -----------------------------
# Save merged JSON
# -----------------------------
print(f"Saving merged laws to {output_file} ...")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(merged_laws, f, ensure_ascii=False, indent=2)

print(f"Stage 3 complete. Total laws: {len(merged_laws)}")
