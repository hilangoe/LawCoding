"""
Stage 3: Merge model predictions with human-coded law JSONs

Output:
- One row per (law_id × key)
- Includes:
    * whether key was human-identified
    * human deontic label
    * whether model identified key
    * aggregated model text
    * predicted deontic label
"""

import json
import requests
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import Dict, List

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent  # project root

DATA_DIR = BASE_DIR / "data"
LAW_JSON_DIR = DATA_DIR / "laws_json"
PRED_PATH = DATA_DIR / "test_predictions.jsonl"
TEST_CSV_PATH = DATA_DIR / "test.csv"

OUT_JSONL = DATA_DIR / "test_data_merged.jsonl"
OUT_CSV = DATA_DIR / "test_data_merged.csv"

# ============================================================
# Load test set (which laws belong in evaluation)
# ============================================================

df_test = pd.read_csv(TEST_CSV_PATH)

law_list = (
    df_test["path"]
      .apply(lambda p: Path(p).stem)  # strip folder + .json
      .dropna()
      .unique()
      .tolist()
)

print("Total laws in test.csv:", len(law_list))
print("Sample law_list IDs:", law_list[:5])

# ============================================================
# Load codebook and rebuild label maps
# ============================================================

CODEBOOK_URL = (
    "https://huggingface.co/spaces/raulpzs/expression_laws/"
    "resolve/main/data3/codebook.json"
)

response = requests.get(CODEBOOK_URL)
response.raise_for_status()
codebook = response.json()


def get_label_maps(codebook: List[Dict]):
    """Rebuild key + deontic label mappings."""
    keys = set()
    for entry in codebook:
        for actor in (entry.get("Actors") or {}).values():
            if actor.get("Key"):
                keys.add(actor["Key"])

    keys = sorted(keys)

    return {
        "key": {
            "id_to_label": {str(i): k for i, k in enumerate(keys)},
            "label_to_id": {k: i for i, k in enumerate(keys)}
        },
        "deontic": {
            "id_to_label": {"0": -1, "1": 1},
            "label_to_id": {-1: 0, 1: 1}
        }
    }


label_maps = get_label_maps(codebook)
key_id_to_label = label_maps["key"]["id_to_label"]
deontic_id_to_label = label_maps["deontic"]["id_to_label"]

# ============================================================
# Load Stage-2 predictions and convert IDs → labels
# ============================================================

stage2_preds = []

with open(PRED_PATH, "r", encoding="utf-8") as f:
    for line in f:
        p = json.loads(line)

        # convert key ID to label
        p["predicted_key_label"] = key_id_to_label.get(
            str(p.get("predicted_key"))
        )

        # convert deontic ID to label
        if p.get("predicted_deontic") is not None:
            p["predicted_deontic_label"] = deontic_id_to_label.get(
                str(p["predicted_deontic"])
            )
        else:
            p["predicted_deontic_label"] = None

        stage2_preds.append(p)

# group predictions by law and key, building type-annotated nested dictionary
# Structure:
# {
#   law_id: {
#       key_label: [prediction_dict, prediction_dict, ...]
#   }
# }
preds_by_law_key: Dict[str, Dict[str, List[Dict]]] = {}

for p in stage2_preds:
    # looping through each prediction in the inference output
    # Example p:
    # {
    #   "law_id": "LAW_001",
    #   "predicted_key_label": "xyx",
    #   "predicted_deontic_label": -1,
    #   "text": "...",
    # }
    # --------------------------------------------

    # extracting law identifier
    law_id = p["law_id"]

    # extracting predicted key (already converted from ID to label earlier)
    key = p["predicted_key_label"]

    # skipping predictions that:
    # 1) are not from a law in our test set
    # 2) do not have a predicted key
    if law_id not in law_list or key is None:
        continue

    # grouping logic:
    # if this law_id does not yet exist in the dictionary, create an empty dictionary for it
    # if this key does not yet exist under that law_id, create an empty list for it
    # then append the ENTIRE prediction dict p
    preds_by_law_key \
        .setdefault(law_id, {}) \
        .setdefault(key, []) \
        .append(p)

# this yields a structure that groups all law predictions by law and key

print("Total law IDs in preds_by_law_key:", len(preds_by_law_key))
print("Sample prediction law IDs:", list(preds_by_law_key.keys())[:5])

# ============================================================
# Load human-coded law JSONs
# ============================================================

human_laws = {}

for law_id in law_list:
    path = LAW_JSON_DIR / f"{law_id}.json"
    if not path.exists():
        continue

    with open(path, "r", encoding="utf-8") as f:
        human_laws[law_id] = json.load(f)

print("Total human laws:", len(human_laws))
print("Sample human law IDs:", list(human_laws.keys())[:5])

# ============================================================
# DEBUG
# ============================================================

missing_from_preds = [law for law in human_laws if law not in preds_by_law_key]
missing_from_human = [law for law in preds_by_law_key if law not in human_laws]

print("Human laws missing from predictions:", missing_from_preds[:10])
print("Prediction laws missing from human:", missing_from_human[:10])

# checking sample law
sample_law = list(human_laws.keys())[0]
human_keys = [p["Provision"] for p in human_laws[sample_law]["Provisions"]]
pred_keys = list(preds_by_law_key.get(sample_law, {}).keys())

print("Human keys:", human_keys)
print("Predicted keys:", pred_keys)

sample_key = human_keys[0]
preds = preds_by_law_key.get(sample_law, {}).get(sample_key, [])
print("Predictions for this key:", preds)


# ============================================================
# Merge per law × key
# ============================================================

# rows will store the final merged data
rows: List[Dict] = []

# example data structures:
# human_laws: Dict[str, Dict] where each Dict has "Provisions" list
# human_laws = {
#     "African_Charter_1981": {
#         "Provisions": [
#             {"Provision": "C_DISINFO_GEN", "Code": "N/A", "Note": None},
#             {"Provision": "C_HUMAN_RIGHTS", "Code": "1", "Note": "Some note"}
#         ]
#     }
# }
#

for law_id, law_data in human_laws.items():  # loop over each law's human-coded JSON
    for prov in law_data["Provisions"]:      # loop over each provision/key
        key = prov["Provision"]             # grab key identifier
        human_code = prov["Code"]           # grab human-coded deontic value

        # human-coded presence & deontic
        human_key_present = human_code != "N/A"  # True if key exists in human coding
        human_deontic = None if human_code == "N/A" else int(human_code)
        # Example: "N/A" -> None, "1" -> 1, "-1" -> -1

        # predicted values for this law-key
        preds = preds_by_law_key.get(law_id, {}).get(key, [])
        # preds is a list of dicts that'll need to get aggregated, each like:
        # {"text": "Some extracted text", "predicted_deontic_label": 1}
        print(f"{law_id} | {key} | #preds={len(preds)}")

        # model-coded presence
        model_key_present = len(preds) > 0   # True if any predictions exist

        # concatenating predicted text for this key, because we might have multiple text snippets labeled with the same key by the model
        # -----------------------------
        predicted_text = (
            "\n\n".join(p["text"].strip() for p in preds)
            if preds else None
        )

        # majority vote on predicted deontic, because there might be contradictory parts of the law
        # generating list of deontics for each key
        deontics = [
            p["predicted_deontic_label"] 
            for p in preds 
            if p["predicted_deontic_label"] is not None
        ]
        # going with the most common deontic value, but in the event of a tie, going deontic==1
        if deontics:
            count_neg1 = deontics.count(-1)
            count_pos1 = deontics.count(1)

            # Rule: if tie, 1 wins
            if count_neg1 > count_pos1:
                predicted_deontic = -1
            else:
                predicted_deontic = 1
        else:
            predicted_deontic = None

        # -----------------------------
        # Append row to final output
        # -----------------------------
        rows.append({
            "law_id": law_id,
            "key": key,

            "human_key_present": human_key_present,
            "human_deontic": human_deontic,

            "model_key_present": model_key_present,
            "predicted_deontic": predicted_deontic,
            "predicted_text": predicted_text,
        })

# After this loop:
# rows is a list of dicts, one per law-key pair, e.g.:
# {
#   'law_id': 'African_Charter_1981',
#   'key': 'C_HUMAN_RIGHTS',
#   'human_key_present': True,
#   'human_deontic': 1,
#   'model_key_present': True,
#   'predicted_deontic': 1,
#   'predicted_text': 'Text extract...'
# }

# ============================================================
# Save outputs
# ============================================================

# JSONL
with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# CSV
pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

print(f"Stage 3 complete.")
print(f"Rows written: {len(rows)}")
print(f"JSONL → {OUT_JSONL}")
print(f"CSV   → {OUT_CSV}")

