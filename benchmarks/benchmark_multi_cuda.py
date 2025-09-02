## DEPRECATED ##

# Global imports
import time
import random
import torch
from multiprocessing import Pool, cpu_count, Queue, Process
import math
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# Moskaengine imports
from research import HandPredictMLP
from moskaengine import MoskaGame, MCTSPlayer, NNMCTSPlayer, HeuristicPlayer

model = None
scaler = None
parent_dir = None
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent

def init_model():
    # Load the model and scaler once per worker
    global model, scaler, parent_dir
    model_path = parent_dir / "models" / "hand_predict_mlp_50k_83epochs.pth"
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

    model_loaded = HandPredictMLP(input_size=485, output_size=156)
    model_loaded.load_state_dict(checkpoint['model_state_dict'])
    model_loaded.eval()

    # Rebuild the scaler with loaded params
    scaler_loaded = MinMaxScaler()
    scaler_loaded.data_min_ = checkpoint['scaler_min']
    scaler_loaded.data_max_ = checkpoint['scaler_max']
    scaler_loaded.scale_ = checkpoint['scaler_scale']
    scaler_loaded.min_ = np.zeros_like(scaler_loaded.scale_)
    scaler_loaded.data_range_ = scaler_loaded.data_max_ - scaler_loaded.data_min_
    scaler_loaded.n_samples_seen_ = np.array([1])  # dummy value, can

    model = model_loaded
    scaler = scaler_loaded

def run_simulation(random_seed):
    global model, scaler

    random.seed(random_seed + random.randint(0, 1000000))
    computer_shuffle = True

    # Cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    players = [HeuristicPlayer('H1'), HeuristicPlayer('H2'), HeuristicPlayer('H3'),
               NNMCTSPlayer("NNMCTS", model, scaler, device, rollouts=100, expl_rate=0.7, scoring="win_rate")]

    game = MoskaGame(players, computer_shuffle, save_vectors=False, print_info=False)

    while not game.is_end_state:
        game.next()

    return str(game.loser).replace('Player ', ''), game.state_data, game.opponent_data

if __name__ == '__main__':
    total_games = 100
    print_every = 1
    start_time = time.time()

    player_names = ['H1', 'H2', 'H3', 'NNMCTS']
    losses = {name: 0 for name in player_names}
    completed = 0

    with Pool(processes=cpu_count(), initializer=init_model) as pool:
        for loser_name, state_data, opponent_data in pool.imap_unordered(run_simulation, range(total_games)):
            losses[loser_name] += 1
            completed += 1

            if completed % print_every == 0:
                print(f"{completed}/{total_games} games completed...")

    # Calculate time and speed
    total_time = time.time() - start_time
    games_per_second = total_games / total_time

    # Output
    print()
    print(f'Total games played: {total_games}')
    print(f'Total simulation time: {total_time:.2f} seconds')
    print(f'Average speed: {games_per_second:.2f} games/second\n')

    # Print loses for each player
    for player, loss_count in losses.items():
        lost_rate = (loss_count / total_games) * 100
        print(f'{player} losses: {loss_count} ({lost_rate:.2f}%)')