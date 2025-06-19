import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from pathlib import Path

# PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Custom imports
from moskaengine.game.deck import StandardDeck

class CardVectorDataset(Dataset):
    def __init__(self, state_file, opponent_file):
        state_table = pq.read_table(state_file)
        opponent_table = pq.read_table(opponent_file)

        self.state_data = state_table.to_pandas().values
        self.opponent_data = opponent_table.to_pandas().values

        if len(self.state_data) != len(self.opponent_data):
            raise ValueError("Mismatch in state/opponent data length")

    def __len__(self):
        return len(self.state_data)

    def __getitem__(self, idx):
        state_vector = torch.from_numpy(self.state_data[idx]).float()
        opponent_vector = torch.from_numpy(self.opponent_data[idx]).float()
        return state_vector, opponent_vector

class HandPredictMLP(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=159):
        super(HandPredictMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()  # Sigmoid for multi-label classification
        )

    def forward(self, x):
        return self.net(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5, device='cpu'):
    model.to(device)
    reference_deck = StandardDeck(shuffle=False, perfect_info=True)
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        all_preds = []
        all_labels = []

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
        with torch.no_grad():
            for val_inputs, val_labels in val_pbar:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)
                total_val_loss += val_loss.item()

                preds = (val_outputs >= 0.5).float()

                all_preds.append(preds.cpu())
                all_labels.append(val_labels.cpu())
                val_pbar.set_postfix(val_loss=val_loss.item())

        avg_val_loss = total_val_loss / len(val_loader)

        # Stack all predictions and labels
        all_preds = torch.vstack(all_preds).numpy()
        all_labels = torch.vstack(all_labels).numpy()

        # Overall metrics
        accuracy = (all_preds == all_labels).sum() / all_labels.size
        precision_micro = precision_score(all_labels, all_preds, average='micro', zero_division=0)
        recall_micro = recall_score(all_labels, all_preds, average='micro', zero_division=0)
        f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)

        precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {accuracy*100:.2f}%")
        print(f"Precision (micro): {precision_micro:.4f} | Recall (micro): {recall_micro:.4f} | F1 (micro): {f1_micro:.4f}")
        print(f"Precision (macro): {precision_macro:.4f} | Recall (macro): {recall_macro:.4f} | F1 (macro): {f1_macro:.4f}")

        # Sample predictions vs labels
        print("Sample predictions vs labels:")

        # Get the first sample's predictions and actual labels
        sample_preds = all_preds[0]  # Get the first row of predictions (0 or 1)
        sample_labels = all_labels[0]  # Get the first row of actual labels (0 or 1)

        # Find predicted hand (only 1's)
        predicted_hand = [card for bit, card in zip(sample_preds, reference_deck) if bit == 1]

        # Find actual hand (only 1's)
        actual_hand = [card for bit, card in zip(sample_labels, reference_deck) if bit == 1]

        print(f"Predicted hand values: {sample_preds, len(sample_preds)}")
        print(f"Predicted hand: {predicted_hand}")
        print(f"Actual hand values: {sample_labels, len(sample_labels)}")
        print(f"Actual hand: {actual_hand}")


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent.parent.parent
    # Define the input and label files
    input_file = parent_dir / "vectors_1k/state/states.parquet"
    label_file = parent_dir / "vectors_1k/opponent/opponents.parquet"

    # Load the dataset
    dataset = CardVectorDataset(input_file, label_file)
    input_size = dataset.state_data.shape[1]

    print(dataset.state_data.shape, dataset.opponent_data.shape)

    # Split dataset
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Detect device (CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model, loss, optimizer
    model = HandPredictMLP(input_size=input_size, output_size=156).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print("Training the model...")
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20)

    # Save the trained model
    model_name = "model_1.pth"
    torch.save(model.state_dict(), model_name)
    print(f"Model saved to: {model_name}")