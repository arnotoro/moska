# Global imports
import time
import uuid
import csv
import random
import torch
import math
from multiprocessing import Pool, cpu_count, Queue, Process
from functools import partial

# Local imports
from moskaengine.game.engine import MoskaGame
# Players
from moskaengine.players.determinized_mcts_player import DeterminizedMCTS
from moskaengine.players.determinized_nn_mcts_player import DeterminizedMLPMCTS
from moskaengine.players.heuristic_player import HeuristicPlayer as Heuristic
# Model
from moskaengine.research.model_training.train_model import HandPredictMLP

def run_simulation(random_seed):
    random.seed(random_seed + random.randint(0, 1000000))
    computer_shuffle = True

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "model_training/model_1.pth"
    model = HandPredictMLP(input_size=433, output_size=156)
    model.load_state_dict(torch.load(model_path, map_location=device))

    players = [Heuristic('H1'), Heuristic('H2'), Heuristic('H3'),
               DeterminizedMLPMCTS("NNMCTS", model, device, rollouts=100, expl_rate=0.7, scoring="win_rate")]

    game = MoskaGame(players, computer_shuffle, save_vectors=False, print_info=True)

    while not game.is_end_state:
        game.next()

    return str(game.loser).replace('Player ', ''), game.state_data, game.opponent_data

if __name__ == '__main__':
    total_games = 1
    print_every = 1
    start_time = time.time()

    player_names = ['H1', 'H2', 'H3', 'NNMCTS']
    losses = {name: 0 for name in player_names}
    completed = 0

    with Pool(processes=cpu_count()) as pool:
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
