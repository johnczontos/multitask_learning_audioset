import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import routes as R
import ontology as O
from data import CachedAudioDataset

train_dataset = CachedAudioDataset(R.train_cache_dir)
# eval_dataset = CachedAudioDataset(R.valid_cache_dir)

# Convert cached datasets to pandas DataFrames
def cached_dataset_to_df(dataset, id_to_name):
    data = []
    for mfcc, labels in dataset:
        label_names = {task: id_to_name[labels[task]] if labels[task] else None for task in labels}
        data.append({'mfcc': mfcc, 'labels': list(labels.values()), 'label_names': list(label_names.values())})
    return pd.DataFrame(data)

train_df = cached_dataset_to_df(train_dataset, O.id_to_name)
# eval_df = cached_dataset_to_df(eval_dataset, O.id_to_name)

print(train_df.info())

# Save basic information about the datasets
def save_basic_info(df, name):
    with open(f"{name}_info.txt", "w+") as f:
        df.info(buf=f)
    description = df.describe(include='all')
    description.to_csv(f"{name}_description.csv")

save_basic_info(train_df, "train_dataset")
# save_basic_info(eval_df, "eval_dataset")

# Plot the distribution of the number of labels per sample
def plot_label_distribution(df, name):
    label_counts = df['labels'].apply(len)
    plt.figure(figsize=(10, 6))
    plt.hist(label_counts, bins=range(1, max(label_counts) + 1), align='left')
    plt.xlabel('Number of Labels per Sample')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Number of Labels per Sample in {name}')
    plt.savefig(f"{name}_label_distribution.png")
    plt.show()

plot_label_distribution(train_df, "Train Dataset")
# plot_label_distribution(eval_df, "Eval Dataset")

# Plot the distribution of human labels
def plot_human_label_distribution(df, name):
    all_human_labels = df['label_names'].explode()
    human_label_counts = all_human_labels.value_counts()
    plt.figure(figsize=(12, 8))
    human_label_counts[:50].plot(kind='bar')
    plt.xlabel('Human Labels')
    plt.ylabel('Frequency')
    plt.title(f'Top 50 Most Frequent Human Labels in {name}')
    plt.tight_layout()
    plt.savefig(f"{name}_human_label_distribution.png")
    plt.show()

plot_human_label_distribution(train_df, "Train Dataset")
# plot_human_label_distribution(eval_df, "Eval Dataset")

# Save a sample of the dataset to a CSV file
def save_sample(df, name, sample_size=100):
    sample_df = df.sample(n=sample_size, random_state=42)
    sample_df.to_csv(f"{name}_sample.csv", index=False)

save_sample(train_df, "train_dataset")
# save_sample(eval_df, "eval_dataset")

print("Data profiling complete. Files saved in the current directory.")
