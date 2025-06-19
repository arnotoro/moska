import random
import time
import torch
from sklearn.preprocessing import MinMaxScaler

from moskaengine.game.engine import MoskaGame
# from moskaengine.players.human_player import Human
from moskaengine.players.random_player import RandomPlayer as Random
from moskaengine.players.determinized_mcts_player import DeterminizedMCTS
from moskaengine.players.heuristic_player import HeuristicPlayer as Heuristic
from moskaengine.players.determinized_nn_mcts_player import DeterminizedMLPMCTS
from moskaengine.research.model_training.train_model import HandPredictMLP

siemen = random.randint(0, 1000000)
# random.seed(siemen)
print(f"Used seed: {siemen}")
random.seed(2539) # For debug

# Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model
model_path = "moskaengine/research/model_training/model_1.pth"
model = HandPredictMLP(input_size=433, output_size=156)
model.load_state_dict(torch.load(model_path, map_location=device))

# Note the main attacker should be specified
# The players can be one of ISMCTS, ISMCTSFPV, DeterminizedMCTS, Random, Human
players = [Random("R1"), Random("R2"), Random("R3"), DeterminizedMCTS('MCTS', deals=1, rollouts=50, expl_rate=0.7, scoring="win_rate")]
# players = [Random('Random'), DeterminizedMCTS('MCTS', deals=5, rollouts=200, expl_rate=0.7, scoring="win_rate")]
# players  = [Random("R1"), Random("R2"), Random("R3"), DeterminizedMLPMCTS('NNMCTS', model, device, deals=1, rollouts=10, expl_rate=0.7, scoring="win_rate")]
# players = [Human('Human1'), Human('Human2')]
# Does the computer shuffle the deck? False for human input used in real-life simulation. Not working yet.
computer_shuffle = True

start = time.time()
game = MoskaGame(players, computer_shuffle, perfect_info=False, save_vectors=True, print_info=True, main_attacker="R1")
while not game.is_end_state:
    game.next()

end = time.time()
print(f"Game took {end - start:.2f} seconds")
# print(game.state_data, game.opponent_data)
print(f'Game is lost by {game.loser}')