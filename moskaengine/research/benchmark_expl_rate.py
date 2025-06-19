from moskaengine.players.determinized_mcts_player import DeterminizedMCTS
from moskaengine.players.random_player import RandomPlayer as Random
from moskaengine.players.determinized_nn_mcts_player import DeterminizedMLPMCTS
from moskaengine.research.model_training.OLD_train_model import CardPredictorMLP
from moskaengine.players.heuristic_player import HeuristicPlayer as Heuristic
from moskaengine.game.engine import MoskaGame
import os
import time
import uuid
import csv
import random
import torch
import math
from multiprocessing import Pool, cpu_count
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
    return str(game.loser).replace('Player ', '')

if __name__ == '__main__':
    total_games = 250
    print_every = 1


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model (if using NN-MCTS)
    model_path = "model_training/card_predictor_1.pth"
    model = CardPredictorMLP(input_size=433, output_size=156)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Exploration rates to test
    expl_rates = [0.05, 0.3, 0.5, 0.7, 1, math.sqrt(2), 2.0, 3.0]

    for expl_rate in expl_rates:
        start_time = time.time()
        print(f"\n=== Testing expl_rate = {expl_rate} ===")

        # Set up players for this run
        players = [
            Heuristic('H1'),
            Heuristic('H2'),
            DeterminizedMCTS('MCTS_reference', deals=3, rollouts=100, expl_rate=1, scoring="win_rate"),
            DeterminizedMCTS('MCTS_test', deals=3, rollouts=100, expl_rate=expl_rate, scoring="win_rate")
        ]

        losses = {player.name: 0 for player in players}

        with Pool(processes=cpu_count()) as pool:
            run_func = partial(run_simulation, players)
            completed = 0

            for loser_name in pool.imap_unordered(run_func, range(total_games)):
                losses[loser_name] += 1
                completed += 1

                if completed % print_every == 0:
                    print(f"{completed}/{total_games} games completed...", end='\r')

        total_time = time.time() - start_time
        games_per_second = total_games / total_time

        print(f"\nResults for expl_rate = {expl_rate}")
        print(f"Total games: {total_games}, Time: {total_time:.2f} s, Speed: {games_per_second:.2f} games/s")

        for player, loss_count in losses.items():
            loserate = (loss_count / total_games) * 100
            print(f'{player} losses: {loss_count} ({loserate:.2f}%)')

