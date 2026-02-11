# LawCoding project

This repository is part of a larger project that's focused on automating the labeling of self-expression laws globally. It includes code for reconstructing training data from human coders (by using LLMs to identify relevant text excerpts), creating synthetic data due to imbalance, and fine-tuning a BERT model for classifying law texts across 300+ variables.

## Pipeline

```mermaid
flowchart LR
    H["Human-coded laws<br>(JSON per law: law_id, provision_key, deontic)"]
    LLM1["GPT-5<br>Provision text extraction<br>(dynamic system instructions)"]
    T1["Reconstructed training data<br>(one row per law-key)"]
    SYN["GPT-5<br>Synthetic data generation<br>(class balancing per key)"]
    T2["Final training dataset<br>(real + synthetic)"]

    AZ1["Azure ML"]
    M1["LEGAL-BERT encoder<br>(frozen)"]
    LORA["LoRA adapters<br>(Q/K/V)"]
    HEADS["Multihead classifier<br>- provision key<br>- deontic"]

    TEST["Human-labeled test laws"]
    SPLIT1["GPT-5-mini<br>Clause-level splitting<br>(strict relevance)"]
    SPLIT2["GPT-5-mini<br>Clause-level splitting<br>(relaxed relevance)"]
    INF1["Inference on clauses<br>(fine-tuned LEGAL-BERT)"]
    INF2["Inference on clauses<br>(fine-tuned LEGAL-BERT)"]

    EVAL1["Evaluation<br>F1 / Precision / Recall<br>(overall & per key)"]
    EVAL2["Evaluation<br>F1 / Precision / Recall<br>(overall & per key)"]

    H --> LLM1 --> T1 --> SYN --> T2
    T2 --> AZ1 --> M1 --> LORA --> HEADS

    TEST --> SPLIT1 --> INF1 --> EVAL1
    TEST --> SPLIT2 --> INF2 --> EVAL2

```