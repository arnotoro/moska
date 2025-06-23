from moskaengine.players.determinized_mcts_player import DeterminizedMCTS
from moskaengine.players.heuristic_player import HeuristicPlayer as Heuristic
from moskaengine.game.engine import MoskaGame
import time
import random
import torch
import math
from multiprocessing import Pool, cpu_count, set_start_method
from functools import partial
import gc

def run_simulation(player_configs, idx):
    print(f"Starting simulation {idx}...\n")
    random.seed(random.randint(0, 1000000))
    computer_shuffle = True

    players = []
    for config in player_configs:
        if config["type"] == "heuristic":
            players.append(Heuristic(config["name"]))
        elif config["type"] == "mcts":
            players.append(DeterminizedMCTS(config["name"],
                                             deals=config["deals"],
                                             rollouts=config["rollouts"],
                                             expl_rate=config["expl_rate"],
                                             scoring=config["scoring"]))

    game = MoskaGame(players,
                     computer_shuffle,
                     save_vectors=False,
                     print_info=False)
    while not game.is_end_state:
        game.next()
    return str(game.loser).replace('Player ', '')

if __name__ == '__main__':
    set_start_method('spawn')
    total_games = 250
    print_every = 1

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    fixed_expl_rate = math.sqrt(2)  # Fixed exploration rate for MCTS
    deals_list = [7]  # Different deals to test
    rollouts_list = [500]

    for deals in deals_list:
        for rollouts in rollouts_list:
            start_time = time.time()
            print(f"\n=== Testing deals={deals}, rollouts={rollouts} ===")

            player_configs = [
                {"type": "heuristic", "name": "H1"},
                {"type": "heuristic", "name": "H2"},
                {"type": "heuristic", "name": "H3"},
                {"type": "mcts", "name": "MCTS_test", "deals": deals, "rollouts": rollouts, "expl_rate": fixed_expl_rate, "scoring": "win_rate"}]

            losses = {config["name"]: 0 for config in player_configs}

            with Pool(processes=cpu_count()) as pool:
                run_func = partial(run_simulation, player_configs)
                completed = 0

                for loser_name in pool.imap_unordered(run_func, range(total_games)):
                    losses[loser_name] += 1
                    completed += 1

                    if completed % print_every == 0:
                        print(f"{completed}/{total_games} games completed...", end='\r')

            # Results
            total_time = time.time() - start_time
            games_per_second = total_games / total_time

            print(f"\nResults for deals={deals}, rollouts={rollouts}")
            print(f"Total games: {total_games}, Time: {total_time:.2f} s, Speed: {games_per_second:.2f} games/s")

            for player, loss_count in losses.items():
                loserate = (loss_count / total_games) * 100
                print(f'{player} losses: {loss_count} ({loserate:.2f}%)')

            # Memory cleanup
            del losses
            del run_func
            torch.cuda.empty_cache()
            gc.collect()