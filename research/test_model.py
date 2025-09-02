import random
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from tqdm import tqdm

# Moskaengine imports
from research.train_model import HandPredictMLP
from moskaengine import StandardDeck

class TestDataset(Dataset):
    def __init__(self, state_data, opponent_data):
        self.state_data = state_data.astype(np.float32)
        self.opponent_data = opponent_data.astype(np.float32)

    def __len__(self):
        return len(self.state_data)

    def __getitem__(self, idx):
        state = torch.tensor(self.state_data[idx], dtype=torch.float32)
        opponent = torch.tensor(self.opponent_data[idx], dtype=torch.float32)
        return state, opponent
    
def evaluate_model(model, test_loader, device="cpu"):
    model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_state, batch_opponent in tqdm(test_loader, desc="Evaluating"):
            batch_state = batch_state.to(device)
            batch_opponent = batch_opponent.to(device)

            logits = model(batch_state)
            probs = torch.sigmoid(logits)

            all_predictions.append(probs.cpu())
            all_labels.append(batch_opponent.cpu())

    return all_predictions, all_labels

def compute_metrics(all_predictions, all_labels, num_opponents=3, num_cards_per_opp=52):

    total_correct = 0
    total_positive = 0
    total_hamming_matches = 0
    total_elements = all_labels.numel()

    for opp in range(num_opponents):
        start = opp * num_cards_per_opp
        end = (opp + 1) * num_cards_per_opp

        labels_opp = all_labels[:, start:end]
        preds_opp = all_predictions[:, start:end]

        # Compute k for each state
        k_per_row = labels_opp.sum(dim=1).int()

        topk_pred_mask = torch.zeros_like(labels_opp)

        for i, k in enumerate(k_per_row):
            if k > 0:
                topk_indices = preds_opp[i].topk(k.item()).indices
                topk_pred_mask[i].scatter_(0, topk_indices, 1)

        # Micro recall
        total_correct += (topk_pred_mask * labels_opp).sum().item()
        total_positive += labels_opp.sum().item()

        # Hamming
        total_hamming_matches += (topk_pred_mask == labels_opp).all(dim=1).sum().item()

    recall_micro = total_correct / total_positive if total_positive > 0 else 0
    hamming_micro = total_hamming_matches / total_elements if total_elements > 0 else 0

    return {
        "recall_micro": recall_micro,
        "hamming_micro": hamming_micro
    }


def compute_topk_metrics(y_pred, y_true, num_opponents=3, num_cards_per_opp=52):
    """
    y_pred: [num_states, num_total_cards] - model probabilities/logits
    y_true: [num_states, num_total_cards] - binary ground truth
    """
    num_states = y_true.shape[0]
    recall_list = []
    hamming_list = []

    for i in range(num_opponents):
        start, end = i * num_cards_per_opp, (i+1) * num_cards_per_opp
        y_true_opp = y_true[:, start:end]
        y_pred_opp = y_pred[:, start:end]

        # Number of cards in this opponent's state per sample
        k_per_sample = y_true_opp.sum(dim=1).long()

        # Compute top-k recall for each sample
        topk_hits = []
        for sample_idx in range(num_states):
            k = k_per_sample[sample_idx].item()
            if k == 0:
                continue  # skip empty hands
            topk_indices = torch.topk(y_pred_opp[sample_idx], k).indices
            true_indices = torch.nonzero(y_true_opp[sample_idx]).flatten()
            hits = len(set(topk_indices.tolist()) & set(true_indices.tolist()))
            topk_hits.append(hits / k)
        recall_list.extend(topk_hits)

        # Compute Hamming accuracy per sample
        hamming_sample = (y_pred_opp.round() == y_true_opp).float().mean(dim=1)
        hamming_list.extend(hamming_sample.tolist())

    recall_micro = sum(recall_list) / len(recall_list)
    hamming_micro = sum(hamming_list) / len(hamming_list)

    return {"recall_micro": recall_micro, "hamming_micro": hamming_micro}

def topk_hamming_accuracy(y_pred, y_true, num_opponents=3, num_cards_per_opp=52):
    """
    y_pred, y_true: [num_states, num_opponents*num_cards_per_opp] tensors
    Returns: top-k Hamming accuracy averaged over all opponents and states
    """
    batch_size = y_true.size(0)
    total_hamming = 0.0
    total_counts = 0

    for i in range(num_opponents):
        start = i * num_cards_per_opp
        end = (i + 1) * num_cards_per_opp

        y_true_opp = y_true[:, start:end]          # [batch_size, 52]
        y_pred_opp = y_pred[:, start:end]          # [batch_size, 52]

        # Number of cards in each state for this opponent
        k = y_true_opp.sum(dim=1).long()           # [batch_size]

        for j in range(batch_size):
            if k[j] == 0:
                continue
            
            topk_vals, topk_idx = torch.topk(y_pred_opp[j], k[j].item())
            pred_topk = torch.zeros_like(y_true_opp[j])
            pred_topk[topk_idx] = 1

            # Hamming accuracy: fraction of matching positions over total positions
            total_hamming += (pred_topk == y_true_opp[j]).sum().item()
            total_counts += y_true_opp[j].numel()

    return total_hamming / total_counts


if __name__ == "__main__":
    # Load model and scaler
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent.parent.parent
    model_path = parent_dir / "models" / "hand_predict_mlp_50k_83epochs.pth"

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

    # Load model weights
    model = HandPredictMLP(input_size=485, output_size=156)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Rebuild the scaler with loaded params
    scaler = MinMaxScaler()
    scaler.data_min_ = checkpoint['scaler_min']
    scaler.data_max_ = checkpoint['scaler_max']
    scaler.scale_ = checkpoint['scaler_scale']
    scaler.min_ = np.zeros_like(scaler.scale_)
    scaler.data_range_ = scaler.data_max_ - scaler.data_min_
    scaler.n_samples_seen_ = np.array([1])  # dummy value, can be any positive integer

    print("Model and scaler loaded successfully.")

    # Load test dataset
    test_dataset_path = parent_dir / "vectors_50k"

    state_df = pq.read_table(test_dataset_path/"state"/"states.parquet").to_pandas()
    label_df = pq.read_table(test_dataset_path/"opponent"/"opponents.parquet").to_pandas()

    state_data = state_df.values.astype(np.float32)
    label_data = label_df.values.astype(np.float32)

    # Test data split
    total_size = len(state_data)
    test_size = int(total_size * 0.1)  # 10% of the data

    test_indices = np.random.choice(total_size, size=test_size, replace=False)

    state_data = state_data[test_indices]
    label_data = label_data[test_indices]

    # Apply scaler
    minmax_cols = list(range(165, 170))
    deck_col = 0  # Deck size is the first column

    state_data[:, minmax_cols] = scaler.transform(state_data[:, minmax_cols])
    state_data[:, deck_col] /= 52.0

    ########################### DEBUG #############################
    print(f"Test dataset shape: {state_data.shape, label_data.shape}")
    
    # Get one row of data randomly from both files
    random_index = random.randint(0, state_data.shape[0] - 1)
    state_row = state_data[random_index]
    opponent_row = label_data[random_index]

    print(f"State row at index {random_index}:\n{state_row}")
    print(f"Opponent row at index {random_index}:\n{opponent_row}")
    ##########################################################
    test_dataset = TestDataset(state_data, label_data)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    print(f"Test dataset shape: {state_data.shape, label_data.shape, test_dataset}")

    # Evaluate model
    all_predictions, all_labels = evaluate_model(model, test_dataloader)

    # Calculate top-k recall averaged over each opponents (micro) and hamming accuracy
    num_opponents = 3
    num_cards_per_opp = 52

    all_labels_tensor = torch.cat(all_labels, dim=0)
    all_predictions_tensor = torch.cat(all_predictions, dim=0)

    metrics = compute_topk_metrics(all_predictions_tensor, all_labels_tensor, num_opponents=num_opponents, num_cards_per_opp=num_cards_per_opp)

    topk_hamming = topk_hamming_accuracy(all_predictions_tensor, all_labels_tensor, num_opponents=num_opponents, num_cards_per_opp=num_cards_per_opp)

    print("Metrics:", metrics)
    print("Top-k Hamming Accuracy:", topk_hamming)