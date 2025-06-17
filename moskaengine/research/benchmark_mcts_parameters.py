from moskaengine.players.determinized_mcts_player import DeterminizedMCTS
from moskaengine.players.random_player import RandomPlayer as Random
from moskaengine.players.heuristic_player import HeuristicPlayer as Heuristic
from moskaengine.game.engine import MoskaGame
import os
import time
import random
import torch
import math
from multiprocessing import Pool, cpu_count
from functools import partial
from scipy.stats import chi2_contingency

def run_simulation(players_list, random_seed):
    random.seed(random_seed + random.randint(0, 1000000))
    computer_shuffle = True
    game = MoskaGame(players_list,
                     computer_shuffle,
                     save_vectors=False,
                     print_info=False)
    while not game.is_end_state:
        game.next()
    return str(game.loser).replace('Player ', '')

from scipy.stats import chi2_contingency

if __name__ == '__main__':
    total_games = 250
    print_every = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    fixed_expl_rate = 1.0
    deals_list = [3]
    rollouts_list = [50, 250, 500, 1000]

    for deals in deals_list:
        for rollouts in rollouts_list:
            start_time = time.time()
            print(f"\n=== Testing deals={deals}, rollouts={rollouts} ===")

            players = [
                Heuristic('H1'),
                Heuristic('H2'),
                DeterminizedMCTS('MCTS_default', deals=3, rollouts=100, expl_rate=fixed_expl_rate, scoring="win_rate"),
                DeterminizedMCTS('MCTS_test', deals=deals, rollouts=rollouts, expl_rate=fixed_expl_rate, scoring="win_rate")
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

            print(f"\nResults for deals={deals}, rollouts={rollouts}")
            print(f"Total games: {total_games}, Time: {total_time:.2f} s, Speed: {games_per_second:.2f} games/s")

            for player, loss_count in losses.items():
                loserate = (loss_count / total_games) * 100
                print(f'{player} losses: {loss_count} ({loserate:.2f}%)')

            # Chi-square significance test between MCTS_default and MCTS_test
            mcts_default_losses = losses['MCTS_default']
            mcts_test_losses = losses['MCTS_test']

            contingency_table = [
                [mcts_default_losses, total_games - mcts_default_losses],
                [mcts_test_losses, total_games - mcts_test_losses]
            ]

            chi2, p_value, dof, expected = chi2_contingency(contingency_table)

            print(f"\nChi-squared test between MCTS_default and MCTS_test:")
            print(f"Chi² = {chi2:.3f}, p-value = {p_value:.4f}")

            if p_value < 0.05:
                print("✅ Statistically significant difference (p < 0.05)")
            else:
                print("❌ No statistically significant difference (p ≥ 0.05)")
