# this is the script for model construction and LoRA wiring
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, PeftModel

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
        out = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        return self.classifier(pooled)

def build_model(base_name, num_keys, num_deontic, lora_cfg, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    
    # 1. Load the base model and move it to the device IMMEDIATELY
    base = AutoModel.from_pretrained(base_name).to(device)

    hidden = base.config.hidden_size
    classifier = MultiHeadClassifier(hidden, num_keys, num_deontic).to(device)

    # 2. Wrap with LoRA while it is already on the target device
    if lora_cfg:
        from peft import get_peft_model, LoraConfig
        base = get_peft_model(base, LoraConfig(**lora_cfg))
    
    # 3. Use a unique attribute name to avoid PEFT internal collisions
    model = CombinedModel(base, classifier)

    return tokenizer, model, hidden

# specific model for using new adapters
def load_trained_model(base_model_name, lora_dir, classifier_ckpt, device=None):
    """
    Load tokenizer + trained LoRA model + classifier for inference
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer + base
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base = AutoModel.from_pretrained(base_model_name).to(device)

    # Load LoRA adapter
    base = PeftModel.from_pretrained(base, lora_dir)

    # Load classifier checkpoint
    ckpt = torch.load(classifier_ckpt, map_location=device)
    classifier = MultiHeadClassifier(
        hidden_size=ckpt["hidden_size"],
        num_keys=ckpt["num_keys"],
        num_deontic=ckpt["num_deontic"]
    ).to(device)
    
    classifier.load_state_dict(ckpt["state_dict"])

    model = CombinedModel(base, classifier)
    model.to(device)
    model.eval()

    return tokenizer, model, device