# LawCoding project

This repository is part of a larger project that's focused on automating the labeling of self-expression laws globally. It includes code for reconstructing training data from human coders (by using LLMs to identify relevant text excerpts), creating synthetic data due to imbalance, and fine-tuning a BERT model for classifying law texts across 300+ variables.

## Pipeline

```mermaid
flowchart LR
    %% Training data creation
    H["Human-coded laws\n(JSON per law: law_id, provision_key, deontic)"]
    
    LLM1["GPT-5\nProvision text extraction\n(dynamic system instructions)"]
    
    T1["Reconstructed training data\n(one row per law-key)"]
    
    SYN["GPT-5\nSynthetic data generation\n(class balancing per key)"]
    
    T2["Final training dataset\n(real + synthetic)"]

    %% Model training
    AZ1["Azure ML"]
    
    M1["LEGAL-BERT encoder\n(frozen)"]
    
    LORA["LoRA adapters\n(Q/K/V)"]
    
    HEADS["Multihead classifier\n- provision key\n- deontic"]

    %% Inference pipeline
    TEST["Human-labeled test laws"]
    
    SPLIT1["GPT-5-mini\nClause-level splitting\n(strict relevance)"]
    SPLIT2["GPT-5-mini\nClause-level splitting\n(relaxed relevance)"]
    
    INF1["Inference on clauses\n(fine-tuned LEGAL-BERT)"]
    INF2["Inference on clauses\n(fine-tuned LEGAL-BERT)"]

    %% Evaluation
    EVAL1["Evaluation\nF1 / Precision / Recall\n(overall & per key)"]
    EVAL2["Evaluation\nF1 / Precision / Recall\n(overall & per key)"]

    %% Edges
    H --> LLM1 --> T1 --> SYN --> T2
    T2 --> AZ1 --> M1 --> LORA --> HEADS

    TEST --> SPLIT1 --> INF1 --> EVAL1
    TEST --> SPLIT2 --> INF2 --> EVAL2

```