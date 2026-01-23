# this is the script for defining the dataset class, creating a dataloader function and a class weights function
import json, torch
from torch.utils.data import Dataset, DataLoader

class LawDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(jsonl_path, "r") as f:
            # filtering out rows with missing labels or empty lines
            all_rows = []
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                # ensuring both labels and text exist before adding to the dataset
                if row.get("key_label") is not None and \
                   row.get("deontic_label") is not None and \
                   row.get("text") is not None:
                    all_rows.append(row)
            
            self.rows = all_rows

        print(f"Loaded {len(self.rows)} valid rows from {jsonl_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        enc = self.tokenizer(
            row["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "key_label": torch.tensor(row["key_label"], dtype=torch.long),
            "deontic_label": torch.tensor(row["deontic_label"], dtype=torch.long),
        }

def make_dataloader(jsonl_path, tokenizer, max_len, batch_size, shuffle=True):
    ds = LawDataset(jsonl_path, tokenizer, max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)

def compute_class_weights(jsonl_path, num_keys):
    counts = [0] * num_keys
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip(): 
                continue 
            row = json.loads(line)
            
            # #### FIX 2: Match the filtering logic of LawDataset exactly ####
            if row.get("key_label") is not None and \
               row.get("deontic_label") is not None and \
               row.get("text") is not None:
                
                label = row["key_label"]
                if label < num_keys: # Safety check for index range
                    counts[label] += 1
            else:
                # We skip silently or log once; LawDataset already warns on load
                pass
    counts = torch.tensor(counts, dtype=torch.float)
    counts[counts == 0] = 1
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_keys
    return weights