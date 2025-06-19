import torch
from moskaengine.research.model_training.train_model import HandPredictMLP
import pandas as pd
from pathlib import Path
from moskaengine.game.deck import StandardDeck
import random

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent.parent

# Load PyTorch model
model_path = "model_1.pth"
model = HandPredictMLP(input_size=433, output_size=156)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Load test dataset
test_dataset_path = parent_dir / "vectors"

# Read file
test_data_state = pd.read_csv(test_dataset_path/"state"/"states.csv")
test_data_opponent = pd.read_csv(test_dataset_path/"opponent"/"opponent.csv")
print(f"Test dataset shape: {test_data_state.shape, test_data_opponent.shape}")

# Print first few rows
print(test_data_state.head())

# Get one row of data randomly from both files
random_index = random.randint(0, test_data_state.shape[0])  # You can change this to any valid index
state_row = test_data_state.iloc[random_index]
opponent_row = test_data_opponent.iloc[random_index]
print(f"State row at index {random_index}:\n{state_row}")
print(f"Opponent row at index {random_index}:\n{opponent_row}")

# Convert to tensors
state_tensor = torch.tensor(state_row.values, dtype=torch.float32)
opponent_tensor = torch.tensor(opponent_row.values, dtype=torch.float32)

# Print tensor shapes
print(f"State tensor shape: {state_tensor.shape}")
print(f"Opponent tensor shape: {opponent_tensor.shape}")

# Calculate number of cards in each opponent's hand
num_opponents = 3
num_cards_per_opponent = 52  # Each opponent can have up to 52 cards
# Create a tensor for the opponent's cards
opponent_tensor = opponent_tensor.view(num_opponents, num_cards_per_opponent)
# Print the reshaped opponent tensor
print(f"Reshaped opponent tensor: {opponent_tensor}")

# Print the number of cards in each opponent's hand
n_cards_opp = []
for i in range(num_opponents):
    num_cards = opponent_tensor[i].sum().item()  # Count the number of cards (1's) in each opponent's hand
    n_cards_opp.append(num_cards)
    print(f"Opponent {i} has {num_cards} cards.")

print(n_cards_opp)

# Predict cards using the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    logits = model(state_tensor.unsqueeze(0))  # Add batch dimension
    probabilities = torch.sigmoid(logits).squeeze()  # Apply sigmoid and remove batch dimension
    print(f"Logits: {logits}")
    print(f"Probabilities: {probabilities}")
    # Divide the probabilities into groups for each opponent
    probabilities = probabilities.view(num_opponents, num_cards_per_opponent)
    print(f"Probabilities reshaped: {probabilities}")
    # Get n_cards_opp number of most probable cards for each opponent
    # This will give us the predicted cards for each opponent
    for i in range(num_opponents):
        top_values, top_indices = torch.topk(probabilities[i], int(n_cards_opp[i]))
        print(f"Top {n_cards_opp[i]} predicted cards for opponent {i}: {top_indices.tolist()}")

        # Mask others to zero
        mask = torch.zeros_like(probabilities[i])

        mask[top_indices] = probabilities[i][top_indices]

        # Convert to binary mask (0 or 1)
        mask = (mask > 0).float()
        probabilities[i] = mask

        print(f"Masked probabilities for opponent {i}: {mask}")

# Print the predicted cards
deck = StandardDeck(shuffle=False, perfect_info=True)

for i in range(3):
    print(f"Predicted cards for opponent {i}:")
    for j in range(52):
        if probabilities[i][j] == 1:
            print(deck.cards[j], end=", ")
    print()  # New line after each opponent's cards

# Correct cards
for i in range(3):
    print(f"Opponent {i} actual cards:")
    for j in range(52):
        if opponent_tensor[i][j] == 1:
            print(deck.cards[j], end=", ")
    print()  # New line after each opponent's cards

# Calculate the value of the predicted cards
predicted_values = []
for i in range(3):
    value = 0
    for j in range(52):
        if probabilities[i][j] == 1:
            value += deck.cards[j].value
    predicted_values.append(value)
# print(f"Predicted values for opponents: {predicted_values}")
# Calculate the value of the actual cards
actual_values = []
for i in range(3):
    value = 0
    for j in range(52):
        if opponent_tensor[i][j] == 1:
            value += deck.cards[j].value
    actual_values.append(value)
# print(f"Actual values for opponents: {actual_values}")
# Calculate the difference between predicted and actual values
for i in range(3):
    diff = predicted_values[i] - actual_values[i]
    print(f"Difference for opponent {i}: {diff}")




