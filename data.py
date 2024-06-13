import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import datasets
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torchaudio.transforms as transforms
import soundfile as sf
import io

import ontology as O
import routes as R

# Define a custom PyTorch dataset class
class AudioMultiTaskDataset(Dataset):
    def __init__(self, dataframe, tasks, target_sample_rate=16000, max_length=160000):
        # Filter the DataFrame to include only relevant samples
        self.dataframe = dataframe[dataframe['labels'].apply(lambda labels: O.name_to_id["Music"] in labels)].reset_index(drop=True)
        self.tasks = tasks
        self.target_sample_rate = target_sample_rate
        self.max_length = max_length
        self.mfcc_transform = transforms.MFCC(
                sample_rate=target_sample_rate,
                n_mfcc=13,
                melkwargs={
                    'n_mels': 40,
                    'n_fft': 400
                }
            )
        self.processed_data = self.preprocess_data()

        # Clean up memory
        del self.dataframe

    def preprocess_data(self):
        processed_data = []

        for idx, row in self.dataframe.iterrows():
            try:
                audio_dict = row['audio']

                # Decode the audio bytes
                audio_bytes = audio_dict['bytes']
                audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))

                audio = torch.tensor(audio_array, dtype=torch.float32)

                # Ensure audio is 1D
                if audio.ndim > 1:
                    audio = audio.mean(dim=1)

                # Downsample if necessary
                if sample_rate != self.target_sample_rate:
                    resample_transform = transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
                    audio = resample_transform(audio)
                    # print("downsampled")
                
                # Ensure audio tensor is in the shape [1, length]
                audio = audio.unsqueeze(0)
                # Cut if too long
                if audio.size(1) > self.max_length:
                    audio = audio[:self.max_length]
                    # print("cut")

                # Pad if too short
                if audio.size(1) < self.max_length:
                    padding = self.max_length - audio.size(0)
                    audio = F.pad(audio, (0, padding))
                    # print("pad")

                # Apply MFCC transformation
                mfcc = self.mfcc_transform(audio)
                # print("mfcc")

                # Extract labels for multiple tasks, use None if task is not present
                labels = {task: None for task in self.tasks}
                for task in self.tasks:
                    for l in row['labels']:
                        if l in self.tasks[task]:
                            labels[task] = l
                            break
                # print(labels)

                processed_data.append((mfcc, labels))
                # print(processed_data)

            except Exception as e:
                print(f"Error processing sample: {idx}\nERROR: {e}")
                break

        return processed_data

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]

# Define the custom collate function for padding
def custom_collate_fn(batch):
    mfccs, labels = zip(*batch)
    # Determine the maximum length of mfccs in the batch
    max_len = max([mfcc.size(2) for mfcc in mfccs])
    # Pad each mfcc to the maximum length
    mfccs_padded = [torch.nn.functional.pad(mfcc, (0, max_len - mfcc.size(2))) for mfcc in mfccs]
    mfccs_padded = torch.stack(mfccs_padded, dim=0)
    return mfccs_padded, labels
    
def cache_dataset(dataset, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for idx, (mfcc, labels) in enumerate(dataset):
        torch.save((mfcc, labels), os.path.join(save_dir, f'{idx}.pt'))

class CachedAudioDataset(Dataset):
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.files = [os.path.join(cache_dir, fname) for fname in os.listdir(cache_dir) if fname.endswith('.pt')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mfcc, labels = torch.load(self.files[idx])
        return mfcc, labels

def dataset_to_pandas_batched(dataset, columns=None, batch_size=100):
    dataframes = []
    for batch in tqdm(dataset.to_pandas(batch_size=batch_size, batched=True), desc="Converting dataset to DataFrame"):
        df = pd.DataFrame(batch)
        if columns:
            df = df[columns]
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

def dataset_to_pandas(dataset, columns=None):
    return dataset.to_pandas()[columns]

if __name__=="__main__":
    # Set datasets logging level
    # datasets.logging.set_verbosity_debug()
    print("loading train...")
    # Load the balanced training dataset
    train_dataset = datasets.load_dataset(
        R.dataset_script_path,
        name="balanced",
        data_dir=R.data_dir,
        split="train",
        cache_dir=R.hf_cache_dir,
        trust_remote_code=True
    )
    print("done.")
    print("loading valid...")
    # # Load the evaluation dataset
    # eval_dataset = datasets.load_dataset(
    #     R.dataset_script_path,
    #     name="balanced",
    #     data_dir=R.data_dir,
    #     split="test",
    #     cache_dir=R.hf_cache_dir,
    #     trust_remote_code=True
    # )
    print("done.")
    print("sample:", train_dataset[0])

    columns_to_keep = ['audio', 'labels']
    train_df = dataset_to_pandas_batched(train_dataset, columns=columns_to_keep)
    # eval_df = dataset_to_pandas(eval_dataset, columns=columns_to_keep)

    print("Creating AudioMultiTaskDataset...")
    train_dataset = AudioMultiTaskDataset(train_df, O.tasks_labels)
    # valid_dataset = AudioMultiTaskDataset(eval_df, O.tasks_labels)
    print("AudioMultiTaskDataset Created.")
    print("sample:", train_dataset[0])

    print("Caching dataset...")
    cache_dataset(train_dataset, R.train_cache_dir)
    # cache_dataset(valid_dataset, R.valid_cache_dir)
    print("Cache complete.")

    # Create DataLoader objects
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    # valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Example of iterating through the DataLoader
    for audio, labels in train_loader:
        print(audio.shape)  # Audio tensor shape
        print(labels)       # Label for the task