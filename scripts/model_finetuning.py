# this the script for fine-tuning our multi-head BERT model
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model
from torch import nn
from tqdm import tqdm

### Configuration
MODEL_NAME = "bert-base-uncased" # example

# Training params
MAX_LEN = 512
BATCH_SIZE = 4
LR = 2e-5
EPOCHS = 4

### Dataset reading, setting LawDataset as subclass of torch.utils.data.Dataset
class LawDataset(Dataset):
    # setting up the data storage
    def __init__(self, jsonl_path, tokenizer):
        self.rows = []
        with open(jsonl_path, "r") as f:
            for line in f:
                self.rows.append(json.loads(line))

        self.tokenizer = tokenizer

    # required for Dataset class
    def __len__(self):
        return len(self.rows)

    # processing rows
    def __getitem__(self, idx):
        row = self.rows[idx]
        # tokenizer creates tenors input_ids and attention_mark
        enc = self.tokenizer(
            row["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            # converting labels into tenors
            "key_label": torch.tensor(row["key_label"], dtype=torch.long),
            "deontic_label": torch.tensor(row["deontic_label"], dtype=torch.long),
        }

### Multihead classifier module 
# subclass of nn.Module (gives automatic parameter tracking, place to define forward() method), submodules. behaves like a standard PyTorch neural network block
class MultiHeadClassifier(nn.Module):
    def __init__(self, hidden_size, num_keys, num_deontic):
        super().__init__()

        # --- Key classifier head (more expressive) ---
        self.key_head = nn.Sequential( # changed from a single layer because it would not be enough to use only one transformation to separate 300 classes
            nn.Linear(hidden_size, hidden_size),  # projection into same-dimensional space
            nn.ReLU(),                            # non-linearity to learn more complex decision boundaries
            nn.Dropout(0.1),                      # regularization
            nn.Linear(hidden_size, num_keys)      # final logits for 300+ classes
        )

        # --- Deontic classifier head (binary) ---
        self.deontic_head = nn.Linear(
            hidden_size, 
            num_deontic
        )

    def forward(self, pooled_output):
        return {
            "key_logits": self.key_head(pooled_output),
            "deontic_logits": self.deontic_head(pooled_output),
        }

### Computing class weights for reweighting because of class imbalance
def compute_class_weights(jsonl_path, num_keys):
    """Compute inverse-frequency class weights for key labels."""
    import json
    counts = [0] * num_keys

    with open(jsonl_path, "r") as f:
        for line in f:
            row = json.loads(line)
            counts[row["key_label"]] += 1

    counts = torch.tensor(counts, dtype=torch.float)

    # avoid division by zero
    counts[counts == 0] = 1

    weights = 1.0 / counts
    weights = weights / weights.sum() * num_keys  # normalize

    return weights

### Combined Model class: creating a subclass for the combined model so it includes forward() method
class CombinedModel(nn.Module):
    def __init__(self, base, classifier):
        super().__init__()
        self.base_model = base
        self.classifier = classifier
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled)

### Training module
def train_model(jsonl_path, metadata_path, output_dir):

    # Load tokenizer + base model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModel.from_pretrained(MODEL_NAME)

    # defining the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Infer hidden size
    hidden_size = base_model.config.hidden_size

    # loading metadata for number of classes
    with open(metadata_path) as f:
        metadata = json.load(f)

    num_keys = metadata["num_keys"]
    num_deontic = metadata["num_deontic"]  # should be 2

    # calculating class weights for keys classes
    class_weights = compute_class_weights(jsonl_path, num_keys)
    class_weights = class_weights.to(device)

    classifier = MultiHeadClassifier(hidden_size, num_keys, num_deontic)

    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1, # increased regularization to avoid overfitting on high-frequency keys
        bias="none",
        target_modules=["query", "key", "value"]  # adding key because of rare classes
    )

    # Apply LoRA to base model
    base_model = get_peft_model(base_model, lora_config)

    # Full model = base + classifier    
    model = CombinedModel(base_model, classifier).to(device)
    
    # printing trainable parameters for LoRA debugging
    model.base_model.print_trainable_parameters()

    dataset = LawDataset(jsonl_path, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    ce_loss = nn.CrossEntropyLoss()
    ce_loss_key = nn.CrossEntropyLoss(weight=class_weights)

    model.to(device)

    # Epoch loop
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:

            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            key_label = batch["key_label"].to(device)
            deontic_label = batch["deontic_label"].to(device)

            # forward through BERT
            outputs = model.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            pooled = outputs.last_hidden_state[:, 0, :] # CLS token

            # forward through classifier
            preds = model.classifier(pooled)

            loss_key = ce_loss_key(preds["key_logits"], key_label)
            loss_deontic = ce_loss(preds["deontic_logits"], deontic_label)
            loss = loss_key + loss_deontic

            loss.backward()
            # allowing gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            pbar.set_postfix({"loss": loss.item()})

    # ensuring dropout is disabled so saved adapter weights are stable
    model.eval()
    
    # Save adapter weights + classifier
    os.makedirs(output_dir, exist_ok=True)
    model.base_model.save_pretrained(os.path.join(output_dir, "lora"))
    torch.save({
        "state_dict": model.classifier.state_dict(),
        "hidden_size": hidden_size,
        "num_keys": num_keys,
        "num_deontic": num_deontic
    }, os.path.join(output_dir, "classifier.pt"))


    print("Training complete. Saved to:", output_dir)

### Entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--meta", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    train_model(args.data, args.meta, args.out)