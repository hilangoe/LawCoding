# this is the run script for inference on the test data (generated locally with API calls)

import json
from pathlib import Path
import torch
from utils_infer import build_model, infer_provisions

# -----------------------------
# Paths & Configs
# -----------------------------
root = Path.cwd()  # repo root (LawCoding/)
CONFIG_DIR = root / "azureml" / "configs"

data_cfg_path = CONFIG_DIR / "data_infer.json"
model_cfg_path = CONFIG_DIR / "model_infer.json"
infer_cfg_path = CONFIG_DIR / "inference.json"
metadata_path = root / "data" / "metadata.json"

# -----------------------------
# Load configs
# -----------------------------
with open(data_cfg_path, "r") as f:
    data_cfg = json.load(f)

with open(model_cfg_path, "r") as f:
    model_cfg = json.load(f)

with open(infer_cfg_path, "r") as f:
    infer_cfg = json.load(f)

with open(metadata_path, "r") as f:
    metadata = json.load(f)

# -----------------------------
# Load test data
# -----------------------------
test_jsonl = Path(data_cfg["test_jsonl"])
provisions = []
with open(test_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        provisions.append(json.loads(line))

print(f"Loaded {len(provisions)} provisions for inference.")

# -----------------------------
# Build model
# -----------------------------
tokenizer, model, _ = build_model(
    base_name=model_cfg["base_model"],
    num_keys=metadata["num_keys"],
    num_deontic=metadata["num_deontic"],
    lora_cfg=model_cfg.get("lora", {})
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()  # inference mode

# -----------------------------
# Run inference
# -----------------------------
batch_size = infer_cfg.get("batch_size", 4)
max_len = infer_cfg.get("max_len", 512)

predictions = infer_provisions(
    provisions=provisions,
    tokenizer=tokenizer,
    model=model,
    batch_size=batch_size,
    max_len=max_len,
    device=device
)

# -----------------------------
# Convert logits to predicted labels
# -----------------------------
for row in predictions:
    key_id = int(torch.tensor(row["key_logits"]).argmax().item())
    row["predicted_key"] = str(key_id)  # string ID to match post-processing

    deontic_id = int(torch.tensor(row["deontic_logits"]).argmax().item())
    row["predicted_deontic"] = deontic_id

print(f"Inference complete. Predicted {len(predictions)} rows.")

# -----------------------------
# Save predictions
# -----------------------------
output_file = Path(__file__).resolve().parent.parent / "data" / "test_predictions.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for row in predictions:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Predictions saved to: {output_file}")


