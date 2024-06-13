import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from model import MultiTaskModel

import routes as R
from data import AudioMultiTaskDataset

train_dataset = load_dataset(
    "agkphysics/AudioSet",
    name="balanced",
    split="train",
    cache_dir=R.cache_dir,
    trust_remote_code=True
)

train_data = AudioMultiTaskDataset(train_dataset)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Initialize model, loss function, and optimizer
input_size = 10  # This should match the actual input size of your audio features
hidden_size = 32
num_layers = 2
num_tasks = len(tasks)

model = MultiTaskModel(input_size, hidden_size, num_layers, num_tasks, num_tasks)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
losses = []
accuracies = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    with tqdm(train_loader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")

            audio, labels = batch
            optimizer.zero_grad()
            outputs = model(audio)
            loss = criterion(outputs, labels.float())

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            total += labels.size(0)
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()

            tepoch.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(train_loader)
    accuracy = 100 * correct / (total * len(tasks))

    losses.append(avg_loss)
    accuracies.append(accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Save the loss and accuracy curves to a CSV file
results_df = pd.DataFrame({
    "epoch": range(1, num_epochs + 1),
    "loss": losses,
    "accuracy": accuracies
})
results_df.to_csv("training_results.csv", index=False)

# Save the trained model
torch.save(model.state_dict(), "multitask_model.pth")

print("Training complete. Model and results saved.")