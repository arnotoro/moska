import random
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class GameStateVectorDataset(Dataset):
    def __init__(self, state, opponent):
        self.state = torch.tensor(state, dtype=torch.float32)
        self.opponent = torch.tensor(opponent, dtype=torch.float32)

    def __len__(self):
        return len(self.state)
    
    def __getitem__(self, idx):
        return self.state[idx], self.opponent[idx]
    
class HandPredictMLP(nn.Module):
    def __init__(self, input_size=485, output_size=156):
        super(HandPredictMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            nn.Sigmoid()  # Sigmoid for multi-label classification
        )

    def forward(self, x):
        return self.net(x)
    
def train_model(model, train_loader, val_loader, criterion, optimizer,
                epochs=5, device='cpu', patience=3, min_delta=1e-4):
    
    # assert device.type == 'cuda', "Device must be CUDA for training"
    model.to(device)

    train_losses = []
    val_losses = []
    val_history = []

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        train_batches = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_batches += 1
            train_pbar.set_postfix(loss=loss.item())

        avg_train_loss = running_train_loss / train_batches
        train_losses.append(avg_train_loss)

        # Validation
        num_opponents = 3
        num_cards_per_opponent = 52

        compute_metrics = (epoch + 1) % 5 == 0 or (epoch + 1) == epochs

        val_loss, val_metrics = evaluate_model(
            model, val_loader, criterion, device,
            num_opponents=num_opponents, num_cards_per_opponent=num_cards_per_opponent,
            compute_metrics=compute_metrics
        )

        val_losses.append(val_loss)
        val_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            **(val_metrics['overall'] if val_metrics else {
                'recall@k': None,
                'precision@k': None,
                'f1@k': None
            })
        })

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_metrics:
            print(f"Overall Recall@k: {val_metrics['overall']['recall@k']:.3f}, "
                  f"F1@k: {val_metrics['overall']['f1@k']:.3f}")

        # Early stopping check
        if best_val_loss - val_loss > min_delta:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter} / {patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                epochs_used = epoch + 1
                break
    else:
        epochs_used = epochs

    return train_losses, val_losses, epochs_used, val_history

def evaluate_model(model, val_loader, criterion, device, num_opponents=3, num_cards_per_opponent=52, compute_metrics=True):
    model.eval()
    running_val_loss = 0.0
    val_batches = 0

    correct_predictions = [[] for _ in range(num_opponents)]
    total_ground_truth_cards = [[] for _ in range(num_opponents)]

    with torch.no_grad():
        for val_inputs, val_labels in tqdm(val_loader, desc="[Val]", leave=False):
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = model(val_inputs)

            val_loss = criterion(val_outputs, val_labels)
            running_val_loss += val_loss.item()
            val_batches += 1
            if compute_metrics:
                val_labels = val_labels.view(-1, num_opponents, num_cards_per_opponent)
                val_outputs = val_outputs.view(-1, num_opponents, num_cards_per_opponent)
                batch_size = val_labels.shape[0]
                n_cards_opp = (val_labels.sum(dim=2)).cpu().numpy()

                top_preds = []
                for i in range(num_opponents):
                    preds_per_opp = []
                    for b in range(batch_size):
                        n_cards = int(n_cards_opp[b, i])
                        if n_cards > 0:
                            _, top_indices = torch.topk(val_outputs[b, i, :], n_cards)
                            preds_per_opp.append(top_indices.cpu().numpy())
                        else:
                            preds_per_opp.append(np.array([]))
                    top_preds.append(preds_per_opp)

                for i in range(num_opponents):
                    for b in range(batch_size):
                        pred_indices = top_preds[i][b]
                        true_indices = (val_labels[b, i, :] == 1).nonzero(as_tuple=True)[0].cpu().numpy()
                        correct = len(set(pred_indices).intersection(set(true_indices)))
                        total = len(true_indices)
                        correct_predictions[i].append(correct)
                        total_ground_truth_cards[i].append(total)

    avg_val_loss = running_val_loss / val_batches

    if not compute_metrics:
        return avg_val_loss, None
    
    # Compute precision/recall/F1 per opponent
    per_opponent_metrics = []
    for i in range(num_opponents):
        total_correct = sum(correct_predictions[i])
        total_true = sum(total_ground_truth_cards[i])
        recall = total_correct / total_true if total_true > 0 else 0.0
        precision = total_correct / total_true if total_true > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_opponent_metrics.append({
            'opponent': i,
            'recall@k': recall,
            'precision@k': precision,
            'f1@k': f1
        })

    # Compute overall metrics
    overall_correct = sum(sum(correct_predictions[i]) for i in range(num_opponents))
    overall_true = sum(sum(total_ground_truth_cards[i]) for i in range(num_opponents))
    overall_recall = overall_correct / overall_true if overall_true > 0 else 0.0
    overall_precision = overall_correct / overall_true if overall_true > 0 else 0.0
    overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)
                if (overall_precision + overall_recall) > 0 else 0.0)

    overall_metrics = {
        'recall@k': overall_recall,
        'precision@k': overall_precision,
        'f1@k': overall_f1
    }

    return avg_val_loss, {
        'overall': overall_metrics,
        'per_opponent': per_opponent_metrics
    }

def test_model_topk_f1(model, test_loader, device='cpu', k_from_ground_truth=True):
    model.to(device)
    model.eval()

    num_opponents = 3
    num_cards_per_opponent = 52
    correct_predictions = [[] for _ in range(num_opponents)]
    total_ground_truth_cards = [[] for _ in range(num_opponents)]

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            labels = labels.view(-1, num_opponents, num_cards_per_opponent)
            outputs = outputs.view(-1, num_opponents, num_cards_per_opponent)
            batch_size = labels.shape[0]
            n_cards_opp = labels.sum(dim=2).cpu().numpy()

            top_preds = []
            for i in range(num_opponents):
                preds_per_opp = []
                for b in range(batch_size):
                    n_cards = int(n_cards_opp[b, i]) if k_from_ground_truth else 6
                    if n_cards > 0:
                        _, top_indices = torch.topk(outputs[b, i, :], n_cards)
                        preds_per_opp.append(top_indices.cpu().numpy())
                    else:
                        preds_per_opp.append(np.array([]))
                top_preds.append(preds_per_opp)

            for i in range(num_opponents):
                for b in range(batch_size):
                    pred_indices = top_preds[i][b]
                    true_indices = (labels[b, i, :] == 1).nonzero(as_tuple=True)[0].cpu().numpy()
                    correct = len(set(pred_indices).intersection(set(true_indices)))
                    total = len(true_indices)
                    correct_predictions[i].append(correct)
                    total_ground_truth_cards[i].append(total)

    for i in range(num_opponents):
        total_correct = sum(correct_predictions[i])
        total_true = sum(total_ground_truth_cards[i])
        recall = total_correct / total_true if total_true > 0 else 0.0
        precision = total_correct / total_true if total_true > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"[Test] Opponent {i}: Recall@k = {recall:.3f}, Precision@k = {precision:.3f}, F1@k = {f1:.3f}")

    overall_correct = sum(sum(correct_predictions[i]) for i in range(num_opponents))
    overall_true = sum(sum(total_ground_truth_cards[i]) for i in range(num_opponents))
    overall_recall = overall_correct / overall_true if overall_true > 0 else 0.0
    print(f"[Test] Overall Recall@k: {overall_recall:.3f}")

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent.parent.parent

    # Load training data
    state_file = parent_dir / "vectors_50k" / "state"/ "states.parquet"
    label_file = parent_dir / "vectors_50k" / "opponent" / "opponents.parquet"

    state_df = pq.read_table(state_file).to_pandas()
    label_df = pq.read_table(label_file).to_pandas()

    state_data = state_df.values.astype(np.float32)
    label_data = label_df.values.astype(np.float32)

    # Define numeric columns for scaling on training data only: deck size, player hand sizes, and game turn
    minmax_cols = list(range(165, 170)) # 166-169 are player hand sizes, 170 is game turn
    deck_col = 0  # Deck size (column 0)

    # Scaler
    scaler = MinMaxScaler()
    scaler.fit(state_data[:int(0.8 * len(state_data)), minmax_cols])
    state_data[:, minmax_cols] = scaler.transform(state_data[:, minmax_cols])
    state_data[:, deck_col] /= 52.0

    # Split ratios
    total_size = len(state_data)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)

    train_indices = np.arange(train_size)
    val_indices = np.arange(train_size, train_size + val_size)
    test_indices = np.arange(train_size + val_size, total_size)

    train_dataset = GameStateVectorDataset(state_data[train_indices], label_data[train_indices])
    val_dataset = GameStateVectorDataset(state_data[val_indices], label_data[val_indices])
    test_dataset = GameStateVectorDataset(state_data[test_indices], label_data[test_indices])

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True, persistent_workers=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, pin_memory=True, persistent_workers=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, pin_memory=True, persistent_workers=True, num_workers=4)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")

    # Print 3 samples from the training data to inspect scaling
    print("Sample scaled training data vectors:\n")
    random_indices = random.sample(range(len(train_dataset)), 3)

    for i, idx in enumerate(random_indices):
        state_vec, label_vec = train_dataset[idx]

        print(f"Sample {i+1} (Index {idx}):")
        print("State vector (first 30 features):", state_vec[:30].numpy())  # First 30 features
        print("State vector (min-max scaled features):", state_vec[minmax_cols].numpy())
        print("State vector (normalized features):", state_vec[deck_col].numpy())
        print("Label vector (first 20):", label_vec[:20].numpy())  # Opponent multi-label vector
        print("-" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model, criterion, and optimizer
    input_size = train_dataset.state.shape[1]
    print(f"Input size: {input_size}, Output size: {train_dataset.opponent.shape[1]}")
    model = HandPredictMLP(input_size=input_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 1000

    train_losses, val_losses, epochs_used, val_history = train_model(model, train_loader, val_loader,
                                        criterion, optimizer,
                                        epochs=epochs,
                                        device=device,
                                        patience=5000, # stop if no improvement in N epochs
                                        min_delta=1e-4 # required improvement to reset patience
                                        )
    
    print(val_history)

    # Plot losses
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Force x-axis to use integer ticks
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Save the plot
    plot_path = parent_dir / "models" / "plots" / f"loss_plot_1k_{epochs_used}epochs.png"
    plt.savefig(plot_path)
    plt.close()

    # Save the trained model
    model_path = parent_dir / "models" / f"hand_predict_mlp_1k_{epochs_used}epochs.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_min': scaler.data_min_,
        'scaler_max': scaler.data_max_,
        'scaler_scale': scaler.scale_,
    }, model_path)
    print(f"Model and scaler saved to {model_path}")

    # Save the validation history
    val_history_path = parent_dir / "models"
    pd.DataFrame(val_history).to_csv(val_history_path / f"val_history_1k_{epochs_used}epochs.csv", index=False)

    # Test the model
    print("\nRunning Top-K F1 evaluation on test set...")
    test_model_topk_f1(model, test_loader, device=device, k_from_ground_truth=True)