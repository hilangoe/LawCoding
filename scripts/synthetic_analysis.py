# script for running analysis on relationship between synthetic data (proportion) and model performance (per key)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from scipy.stats import linregress

# loading the per-provision metrics dataset from v2 of inference pipeline
metrics_path = "../data/analysis/key_frequency_agreement_v2.csv"

df_metrics = pd.read_csv(metrics_path)

print(df_metrics.head())

# loading training data

training_path = "../data/training_data.jsonl"

with open(training_path, 'r') as f:
    df = pd.DataFrame([json.loads(line) for line in f])

# generating synthetic variable
df_training = (
    df.groupby('key')['law_id']
      .apply(lambda x: (x == 'synth').mean())
      .rename('prop_synth')
      .reset_index()
)

print(df_training.head())

print(df_training.describe())

# joining the two dfs
df_merged = df_metrics.merge(df_training, on="key", how="left")

print(df_merged.head())

print(df_merged.describe())


# generating f1 score
df_merged['f1'] = np.where(
    (df_merged['precision'] + df_merged['recall']) > 0,
    2 * df_merged['precision'] * df_merged['recall']
    / (df_merged['precision'] + df_merged['recall']),
    0
)

print(df_merged.describe())

# metrics to plot
metrics = ['f1', 'precision', 'recall']


for metric in metrics:
    # Prepare data
    plot_df_metric = df_merged[['prop_synth', metric]].dropna()

    # Linear regression for p-value
    res = linregress(plot_df_metric['prop_synth'], plot_df_metric[metric])
    pvalue = res.pvalue
    print(f"{metric.upper()} vs prop_synth p-value: {pvalue:.3g}")

    # Create plot
    plt.figure(figsize=(7,5))
    ax = sns.regplot(
        data=plot_df_metric,
        x="prop_synth",
        y=metric,
        ci=95,
        scatter_kws={"alpha":0.7},
        line_kws={"color":"red", "linewidth":2}
    )

    # Annotate p-value
    plt.text(
        0.05, 0.95,
        f"p = {pvalue:.3g}",
        transform=ax.transAxes,
        verticalalignment='top'
    )

    plt.xlabel("Proportion of synthetic training data")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} vs proportion of synthetic data")
    plt.tight_layout()

    # Save each figure
    plt.savefig(f"../outputs/synth_{metric}_scatter.png", dpi=300)
    plt.close()

# sanity check
# filter rows where both prop_synth and f1 are greater than 0
subset = df_merged[(df_merged['prop_synth'] > 0) & (df_merged['f1'] > 0)]

print(f"Rows where both prop_synth and f1 are greater than zero: {subset}")

# precision
subset = df_merged[(df_merged['prop_synth'] > 0) & (df_merged['precision'] > 0)]

print(f"Rows where both prop_synth and precision are greater than zero: {subset}")

# recall
subset = df_merged[(df_merged['prop_synth'] > 0) & (df_merged['recall'] > 0)]

print(f"Rows where both prop_synth and recall are greater than zero: {subset}")