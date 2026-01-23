# this is the library module for the inference script on Azure, conducting stage 2

import json
from typing import List, Dict

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

# ---------------------------------------------------------------------
# Model definitions (reused from training)
# ---------------------------------------------------------------------

class MultiHeadClassifier(nn.Module):
    def __init__(self, hidden_size, num_keys, num_deontic):
        super().__init__()
        self.key_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_keys)
        )
        self.deontic_head = nn.Linear(hidden_size, num_deontic)

    def forward(self, pooled):
        return {
            "key_logits": self.key_head(pooled),
            "deontic_logits": self.deontic_head(pooled),
        }


class CombinedModel(nn.Module):
    def __init__(self, base, classifier):
        super().__init__()
        self.base_model = base
        self.classifier = classifier

    def forward(self, input_ids, attention_mask):
        out = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = out.last_hidden_state[:, 0, :]  # CLS token
        return self.classifier(pooled)

# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

# function for loading the model
def load_trained_model(
    base_model_name: str,
    lora_dir: str,
    classifier_ckpt: str,
    device: str | None = None,
):
    """
    Load tokenizer + trained model for inference.

    Parameters
    ----------
    base_model_name : str
        HuggingFace base model (e.g. bert-base-uncased)

    lora_dir : str
        Path to saved LoRA adapter directory

    classifier_ckpt : str
        Path to classifier.pt checkpoint

    device : str, optional
        "cuda" or "cpu"
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # grabbing tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base = AutoModel.from_pretrained(base_model_name)

    # loading trained LoRA adapters
    base = PeftModel.from_pretrained(base, lora_dir)

    ckpt = torch.load(classifier_ckpt, map_location=device)

    # in case the base model has not changed from training
    hidden = base.config.hidden_size
    assert hidden == ckpt["hidden_size"], "Base model hidden size does not match classifier checkpoint"

    
    # defining the classifier
    classifier = MultiHeadClassifier(
        hidden_size=hidden,
        num_keys=ckpt["num_keys"],
        num_deontic=ckpt["num_deontic"]
    )
    classifier.load_state_dict(ckpt["state_dict"])

    model = CombinedModel(base, classifier)
    model.to(device)
    model.eval()

    return tokenizer, model, device

# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------


@torch.no_grad()
def infer_provisions(
    provisions: List[Dict],
    tokenizer,
    model,
    device,
    max_len: int = 512,
    batch_size: int = 8,
):
    """
    Run inference on provision texts.

    Parameters
    ----------
    provisions : list[dict]
        Each dict must contain at least:
            { "text": "...", "law_id": "...", ... }

    Returns
    -------
    list[dict]
        One output row per provision with raw logits
    """

    outputs = []

    for i in range(0, len(provisions), batch_size):
        batch = provisions[i : i + batch_size]
        texts = [p["text"] for p in batch]

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )

        enc = {k: v.to(device) for k, v in enc.items()}

        preds = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"]
        )

        key_logits = preds["key_logits"].cpu()
        deontic_logits = preds["deontic_logits"].cpu()

        for j, prov in enumerate(batch):
            outputs.append({
                "law_id": prov.get("law_id"),
                "text": prov.get("text"),
                "key_logits": key_logits[j].tolist(),
                "deontic_logits": deontic_logits[j].tolist(),
            })

    return outputs