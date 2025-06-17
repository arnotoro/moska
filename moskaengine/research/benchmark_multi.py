from moskaengine.players.determinized_mcts_player import DeterminizedMCTS
from moskaengine.players.random_player import RandomPlayer as Random
from moskaengine.players.determinized_mcts_player import DeterminizedMCTS
from moskaengine.players.determinized_nn_mcts_player import DeterminizedMLPMCTS
from moskaengine.research.model_training.train_model import CardPredictorMLP
from moskaengine.players.heuristic_player import HeuristicPlayer as Heuristic
from moskaengine.game.engine import MoskaGame
import os
import time
import uuid
import csv
import random
import torch
import math
from multiprocessing import Pool, cpu_count, Queue, Process
from functools import partial

def run_simulation(players_list, random_seed):
    random.seed(random_seed + random.randint(0, 1000000))
    computer_shuffle = True
    game = MoskaGame(players_list,
                     computer_shuffle,
                     save_vectors=False,
                        print_info=False,
                     )
    while not game.is_end_state:
        game.next()
    return str(game.loser).replace('Player ', ''), game.state_data, game.opponent_data

if __name__ == '__main__':
    total_games = 5000
    print_every = 100
    start_time = time.time()

    # Cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    model_path = "model_training/card_predictor_1.pth"
    model = CardPredictorMLP(input_size=433, output_size=156)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # players = [Heuristic('Heuristic1'), Heuristic('Heuristic2'), Heuristic('Heuristic3'), DeterminizedMLPMCTS('NNMCTS', model, device, deals=3, rollouts=100, expl_rate=0.7, scoring="win_rate")]
    players = [Heuristic('H1'), Heuristic('H2'), DeterminizedMCTS('MCTS_reference', deals=5, rollouts=250, expl_rate=0.7, scoring="win_rate"), Heuristic('H3')]
    losses = {player.name: 0 for player in players}  # or however many players
    batch_number = 1

    # Get the directory for saving the game vectors

    with Pool(processes=cpu_count()) as pool:
        completed = 0
        run_func = partial(run_simulation, players)

        for loser_name, state_data, opponent_data in pool.imap_unordered(run_func, range(total_games)):
            losses[loser_name] += 1
            completed += 1

            if completed % print_every == 0:
                batch_number += 1
                print(f"{completed}/{total_games} games completed...")

    # Calculate time and speed
    total_time = time.time() - start_time
    games_per_second = total_games / total_time

    # Output
    print()
    print(f'Total games played: {total_games}')
    print(f'Total simulation time: {total_time:.2f} seconds')
    print(f'Average speed: {games_per_second:.2f} games/second\n')

    # Print loserates
    for player, loss_count in losses.items():
        loserate = (loss_count / total_games) * 100
        print(f'{player} losses: {loss_count} ({loserate:.2f}%)')
