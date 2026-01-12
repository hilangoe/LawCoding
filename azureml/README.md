# Azure ML Job README — BERT Adapter Fine-Tuning
Overview

This project contains the Azure ML job definitions and configurations for fine-tuning a BERT-based model with 
adapters on legal text datasets related to self-expression laws globally (focused on dis- and misinformation). The 
training data contains a mix of human-coded labels and synthetic data due to the imbalance in provision keys in the 
former.

We use two main workflows:

Development workflow — small dataset, short training for fast testing on the Compute Instance (CI).

Full training workflow — full dataset, full training configuration, executed on the Compute Cluster.
