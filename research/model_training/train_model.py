import torch
import torch.nn as nn
import pyarrow.parquet as pq
from torch.utils.data import Dataset, random_split, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

from moskaengine.game.deck import StandardDeck

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
    def __init__(self, input_size, hidden_size=128, output_size=156):
        super(CardPredictorMLP, self).__init__()
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

def load_and_prepare_data(game_state_file, label_file=None, test_size=0.2, random_seed=69):
    # Load data
    state_data = pq.read_table(game_state_file)
    game_states = state_data.to_pandas().values

    if label_file:
        label_data = pq.read_table(label_file)
        labels = label_data.to_pandas().values

        if len(game_states) != len(labels):
            raise ValueError("Mismatch in game state and label data length")

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            game_states, labels, test_size=test_size, random_state=random_seed
        )

        # Create datasets
        print(X_train.shape, y_train.shape)
        train_dataset = GameDataset(X_train, y_train)
        test_dataset = GameDataset(X_test, y_test)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        return train_loader, test_loader
    else:
        # If no labels, just create a dataset for prediction
        game_state_dataset = GameDataset(game_states)
        game_data_loader = DataLoader(game_state_dataset, batch_size=32, shuffle=False)
        return game_data_loader

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
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()

            # For metrics
            total += targets.size(0) * targets.size(1)
            correct += (predicted == targets).sum().item()

            all_predictions.append(predicted.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f"Test accuracy: {accuracy:.2f}%")

    # Calculate precision, recall, and F1 score


    return accuracy


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
    game_state_file = "../../vectors_10k/state/states_10k.parquet"
    opponent_file = "../../vectors_10k/opponent/opponents_10k.parquet"

    # Load data
    train_loader, test_loader = load_and_prepare_data(game_state_file, opponent_file)

    # Check if data is loaded correctly
    print(f"Number of training samples: {len(train_loader.dataset[0])}")

    # Initialize model
    # model = CardPredictorCNN(input_size=433, output_size=156)
    model = CardPredictorMLP(input_size=433, hidden_size=128, output_size=156)
    # print(model)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Train the model
    print("Training the model...")
    train_model(model, train_loader, criterion, optimizer, epochs=5, device=device)

    # Save the trained model
    model_name = "card_predictor_cnn_1.pth"
    torch.save(model.state_dict(), model_name)
    print(f"Model saved to: {model_name}")

    # Final evaluation
    print("Final evaluation on test set:")
    evaluate(model, test_loader, device=device)


