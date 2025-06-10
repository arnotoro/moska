import torch
import torch.nn as nn
import pyarrow.parquet as pq
from torch.utils.data import Dataset, random_split, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm

class CardPredictorCNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(CardPredictorCNN, self).__init__()

        # CNN blocks
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
        )

        # Size of flattened features after CNN layers
        self.flattened_size = 512 * (input_size // 4)  # Adjust based on pooling

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_size),
            nn.Sigmoid()  # Sigmoid for multi-label classification
        )

    def forward(self, x):
        # Reshape input for CNN
        x = x.unsqueeze(1)

        # Pass through CNN layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        x = self.fc(x)

        return x

class CardPredictorMLP(nn.Module):
    def __init__(self, input_size, output_size=156):
        super(CardPredictorMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            nn.Sigmoid()  # Sigmoid for multi-label classification
        )

    def forward(self, x):
        return self.net(x)

class GameDataset(Dataset):
    def __init__(self, game_states, labels):
        self.game_states = game_states
        self.labels = labels

    def __len__(self):
        return len(self.game_states)

    def __getitem__(self, idx):
        state_vector = torch.from_numpy(self.game_states[idx]).float()
        label_vector = torch.from_numpy(self.labels[idx]).float()
        return state_vector, label_vector

def get_dataset(state_file, label_file=None, test_size=0.2, random_state=69):
    # Load data files
    X = pq.read_table(state_file)
    state_data = X.to_pandas()

    # find columns that are not binary (0 or 1)
    non_binary_columns = [col for col in state_data.columns if not state_data[col].isin([0, 1]).all()]

    print(f"Number of non-binary columns: {non_binary_columns}")

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    state_data[non_binary_columns] = scaler.fit_transform(state_data[non_binary_columns])

    if label_file:
        # If a label file is provided, we load the labels
        Y = pq.read_table(label_file)
        label_data = Y.to_pandas().values

        state_data = state_data.values

        assert state_data.shape[0] == label_data.shape[0], "State and label data must have the same number of samples."

        # Split the dataset into training and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(
            state_data, label_data, test_size=test_size, random_state=random_state
        )

        train_dataset = GameDataset(X_train, Y_train)
        test_dataset = GameDataset(X_test, Y_test)

        # Create DataLoaders for training and test sets
        train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Return the DataLoader for training
        return train_data_loader, test_data_loader
    else:
        # If no label file is provided, we create a dataset for prediction
        state_data = GameDataset(state_data)
        state_data_loader = DataLoader(state_data, batch_size=64, shuffle=False)
        return state_data_loader

def train_model(model, train_loader, criterion, optimizer, epochs=5, device='cpu'):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)

        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            # print(inputs.shape, targets.shape)
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update statistics
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            batch_total = targets.size(0) * targets.size(1)
            total += batch_total
            batch_correct = (predicted == targets).sum().item()
            correct += batch_correct

            # Update progress bar with current batch statistics
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'batch_acc': f'{100 * batch_correct / batch_total:.2f}%',
                'avg_loss': f'{running_loss / (train_pbar.n + 1):.4f}',
                'avg_acc': f'{100 * correct / total:.2f}%'
            })

        # Close the progress bar
        train_pbar.close()

        # Print epoch summary
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

        # Evaluate on test set after each epoch
        if test_loader is not None:
            evaluate(model, test_loader, device)

def evaluate(model, test_loader, device='cpu'):
    model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()

            all_predictions.append(predicted.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    # Compute metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision_macro = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
    recall_macro = recall_score(all_targets, all_predictions, average='macro', zero_division=0)
    f1_macro = f1_score(all_targets, all_predictions, average='macro', zero_division=0)

    precision_micro = precision_score(all_targets, all_predictions, average='micro', zero_division=0)
    recall_micro = recall_score(all_targets, all_predictions, average='micro', zero_division=0)
    f1_micro = f1_score(all_targets, all_predictions, average='micro', zero_division=0)

    print(f"\n📊 Evaluation Metrics:")
    print(f"Accuracy:         {accuracy * 100:.2f}%")
    print(f"Macro Precision:  {precision_macro:.4f}")
    print(f"Macro Recall:     {recall_macro:.4f}")
    print(f"Macro F1 Score:   {f1_macro:.4f}")
    print(f"Micro Precision:  {precision_micro:.4f}")
    print(f"Micro Recall:     {recall_micro:.4f}")
    print(f"Micro F1 Score:   {f1_micro:.4f}\n")

    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro
    }


def predict(model, data_loader, device='cpu'):
    model.to(device)
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for inputs in data_loader:
            if isinstance(inputs, tuple):  # If the dataloader returns a tuple (inputs, targets)
                inputs = inputs[0]

            inputs = inputs.to(device)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            all_predictions.append(predicted.cpu().numpy())

    return np.vstack(all_predictions)

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # File paths
    game_state_file = "../../vectors_10k/state/states.parquet"
    opponent_file = "../../vectors_10k/opponent/opponents.parquet"

    # Load data
    train_loader, test_loader = get_dataset(game_state_file, opponent_file)

    # Check if data is loaded correctly
    print(f"Number of training samples: {len(train_loader.dataset)}")

    # Initialize model
    model = CardPredictorMLP(input_size=433, output_size=156)
    model.to(device)
    print(model)

    # 📊 Compute positive counts per label from train_loader
    print("Computing positive label counts...")
    num_classes = 156
    pos_counts = torch.zeros(num_classes).to(device)

    total_samples = 0
    for _, labels in train_loader:
        labels = labels.to(device)
        pos_counts += labels.sum(dim=0)
        total_samples += labels.size(0)

    neg_counts = total_samples - pos_counts
    pos_weight_values = neg_counts / (pos_counts + 1e-6)  # prevent division by zero
    pos_weight_tensor = pos_weight_values.to(device)

    print(pos_counts, total_samples, neg_counts)
    print(f"Positive counts per class: {pos_counts}")
    print(f"Positive weights: {pos_weight_tensor}")

    # Loss and optimizer with pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Train the model
    print("Training the model...")
    train_model(model, train_loader, criterion, optimizer, epochs=5, device=device)

    # Save the trained model
    model_name = "card_predictor_1.pth"
    torch.save(model.state_dict(), model_name)
    print(f"Model saved to: {model_name}")


