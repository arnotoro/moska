from sympy.printing.pytorch import torch

from moskaengine.research.model_training.OLD_train_model import CardPredictorCNN, evaluate, load_and_prepare_data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

# Load the model structure first
model = CardPredictorCNN(input_size=433, output_size=156)
model.to(device)


# File paths
game_state_file = "../../../vectors_10k/state/states.parquet"
opponent_file = "../../../vectors_10k/opponent/opponents.parquet"

# Load the saved weights
model.load_state_dict(torch.load("card_predictor_cnn_1.pth", map_location=device))
print("Model loaded.")
model.eval()  # Set to eval mode

# Load data again (or reuse your test_loader if still available)
_, test_loader = load_and_prepare_data(game_state_file, opponent_file)

# Run evaluation
evaluate(model, test_loader, device=device)
