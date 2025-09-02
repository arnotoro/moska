import random
import time
import torch
from sklearn.preprocessing import MinMaxScaler
import math
import random
import time
import torch
from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np

# Moskaengine imports
from research import HandPredictMLP
from moskaengine import MoskaGame, HumanPlayer, RandomPlayer, MCTSPlayer, HeuristicPlayer, NNMCTSPlayer

siemen = random.randint(0, 1000000)
# random.seed(siemen)
print(f"Used seed: {siemen}")
random.seed(siemen) # For debug

# # Cuda
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Load the model for DeterminizedMLPMCTS
# model_path = "models/hand_predict_mlp_25k_95epochs.pth"
# checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

# # Load model weights
# model = HandPredictMLP(input_size=485, output_size=156)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()

# # Rebuild the scaler with loaded params
# scaler = MinMaxScaler()
# scaler.data_min_ = checkpoint['scaler_min']
# scaler.data_max_ = checkpoint['scaler_max']
# scaler.scale_ = checkpoint['scaler_scale']

# # For MinMaxScaler, set these attributes:
# scaler.min_ = np.zeros_like(scaler.scale_)
# scaler.data_range_ = scaler.data_max_ - scaler.data_min_
# scaler.n_samples_seen_ = np.array([1])  # dummy value, can be any positive integer

# print("Model and scaler loaded successfully.")


players = [HumanPlayer('me'), HeuristicPlayer('H2'), HeuristicPlayer('H3'), MCTSPlayer('MCTS', deals=10, rollouts=750, expl_rate=1.7, scoring="win_rate")]
# players = [HumanPlayer('me'), HeuristicPlayer('H2'), HeuristicPlayer('H3'), DeterminizedMLPMCTS('NNMCTS', model, scaler, device, deals=10, rollouts=750, expl_rate=math.sqrt(2), scoring="win_rate")]

computer_shuffle = True

start = time.time()
game = MoskaGame(players, computer_shuffle, perfect_info=False, save_vectors=False, print_info=False)
while not game.is_end_state:
    game.next()

end = time.time()
print(f"Game took {end - start:.2f} seconds")
# print(game.state_data, game.opponent_data)
print(f'Game is lost by {game.loser}')