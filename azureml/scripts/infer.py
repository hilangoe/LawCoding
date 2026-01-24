# this is the run script for inference on the test data (generated locally with API calls)
import json
from pathlib import Path
import argparse
import torch
import os
from utils_infer import load_trained_model, infer_provisions

# -----------------------------
# Parse arguments
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_weights_path", type=str, help="Path to adapters folder")
parser.add_argument("--data_config", type=str, required=True)
parser.add_argument("--model_config", type=str, required=True)
parser.add_argument("--inference_config", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

# -----------------------------
# Configs
# -----------------------------
with open(args.data_config, "r") as f:
    data_cfg = json.load(f)

with open(args.model_config, "r") as f:
    model_cfg = json.load(f)

with open(args.inference_config, "r") as f:
    infer_cfg = json.load(f)

batch_size = infer_cfg.get("batch_size", 4)
max_len = infer_cfg.get("max_len", 512)

# -----------------------------
# Loading test data
# -----------------------------
test_jsonl = Path(data_cfg["test_jsonl"])
provisions = []
with open(test_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        provisions.append(json.loads(line))

print(f"Loaded {len(provisions)} provisions for inference.")

# -----------------------------
# Loading trained model (BASE + LoRA + CLASSIFIER)
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# #### new directory structure created during training.
lora_path = os.path.join(args.model_weights_path, "lora")
classifier_path = os.path.join(args.model_weights_path, "classifier.pt")

# -----------------------------
# DEBUG: Check that input files exist
# -----------------------------
print(f"Debug: model_weights_path = {args.model_weights_path}")
print(f"Debug: classifier_path = {classifier_path}")
print(f"Debug: lora_path = {lora_path}")

if not os.path.isfile(classifier_path):
    print("WARNING: classifier.pt not found!")

if not os.path.isdir(lora_path):
    print("WARNING: lora folder not found!")
else:
    lora_files = os.listdir(lora_path)
    print(f"Found {len(lora_files)} files in lora folder: {lora_files}")

tokenizer, model, _ = load_trained_model(
    base_model_name=model_cfg["base_model"],
    lora_dir=lora_path,
    classifier_ckpt=classifier_path,
    device=device
)

# -----------------------------
# Running inference
# -----------------------------
# added model.eval() and torch.no_grad() to ensure weights aren't updated and memory is optimized
model.eval()
with torch.no_grad():
    predictions = infer_provisions(
        provisions=provisions,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_len=max_len
    )

# -----------------------------
# Converting logits to predicted labels
# -----------------------------
for row in predictions:
    # added .cpu() before .argmax() to ensure safety if running on GPU.
    # using torch.as_tensor() because it is more efficient since data is already a list/array.
    row["predicted_key"] = int(
        torch.as_tensor(row["key_logits"]).cpu().argmax().item()
    )
    row["predicted_deontic"] = int(
        torch.as_tensor(row["deontic_logits"]).cpu().argmax().item()
    )

print(f"Inference complete. Predicted {len(predictions)} rows.")

# -----------------------------
# Saving predictions
# -----------------------------
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

output_file = output_dir / "test_predictions.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for row in predictions:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Predictions saved to: {output_file}")