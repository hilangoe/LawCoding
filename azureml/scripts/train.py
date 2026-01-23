# this is the script for running the fine-tuning
import os
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# local modules
from model import build_model
from data import make_dataloader, LawDataset, compute_class_weights
from utils import set_seed

# loading data safely on azure and locally

from pathlib import Path

def train_model(data_cfg_path, training_cfg_path, model_cfg_path, output_dir):

    root = Path.cwd()
    
    # loading config files
    with open(data_cfg_path) as f:
        data_cfg = json.load(f)
    with open(training_cfg_path) as f:
        train_cfg = json.load(f)
    with open(model_cfg_path) as f:
        model_cfg = json.load(f)

    metadata_path = root / data_cfg["metadata_path"]

    # loading meta data
    with open(metadata_path) as f:
        metadata = json.load(f)
    num_keys = metadata["num_keys"]
    num_deontic = metadata["num_deontic"]

    SAVE_DIR = Path(output_dir)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # setting training parameters
    MAX_LEN = train_cfg["max_len"]
    BATCH_SIZE = train_cfg["batch_size"]
    LR = train_cfg["learning_rate"]
    EPOCHS = train_cfg["epochs"]
    GRAD_CLIP = train_cfg.get("gradient_clip", 1.0)

    BASE_MODEL = model_cfg["base_model"]
    USE_LORA = model_cfg.get("use_lora", True)
    LORA_CFG = model_cfg.get("lora", {})

    set_seed(train_cfg.get("seed", 42))

    # building models
    tokenizer, model, hidden = build_model(
        base_name=BASE_MODEL,
        num_keys=num_keys,
        num_deontic=num_deontic,
        lora_cfg=LORA_CFG if USE_LORA else {}
    )

    # preparing dataset and loader
    train_file = root / data_cfg["train_jsonl"]

    loader = make_dataloader(train_file, tokenizer, MAX_LEN, BATCH_SIZE, shuffle=True)

    # calculating class weights
    class_weights = compute_class_weights(train_file, num_keys).to("cuda" if torch.cuda.is_available() else "cpu")

    # training setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # debug check
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    
    ce_loss = nn.CrossEntropyLoss()
    ce_loss_key = nn.CrossEntropyLoss(weight=class_weights)

    # training loop
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            key_label = batch["key_label"].to(device)
            deontic_label = batch["deontic_label"].to(device)

            # forward
            preds = model(input_ids=input_ids, attention_mask=attention_mask)

            # loss
            loss_key = ce_loss_key(preds["key_logits"], key_label)
            loss_deontic = ce_loss(preds["deontic_logits"], deontic_label)
            loss = loss_key + loss_deontic

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            pbar.set_postfix({"loss": loss.item()})

    # saving model
    if USE_LORA:
        # saving LoRA adapter separately
        model.base_model.save_pretrained(os.path.join(SAVE_DIR, "lora"))
    
    torch.save(
        {
            "state_dict": model.classifier.state_dict(),
            "hidden_size": hidden,
            "num_keys": num_keys,
            "num_deontic": num_deontic
        },
        os.path.join(SAVE_DIR, "classifier.pt")
    )

    print("Training complete. Saved to:", SAVE_DIR)


# entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", type=str, required=True)
    parser.add_argument("--training_config", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    train_model(args.data_config, 
                args.training_config, 
                args.model_config,
               args.output_dir
    )

