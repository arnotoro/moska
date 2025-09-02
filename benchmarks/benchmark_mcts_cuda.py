import time
import random
import os
import gc
import traceback
import math
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import cpu_count, get_context, TimeoutError
from functools import partial

# Moskaengine imports
from research import HandPredictMLP
from moskaengine import MoskaGame, HeuristicPlayer, MCTSPlayer, NNMCTSPlayer

def run_simulation(player_configs, idx):
    try:
        print(f"Starting simulation {idx}...\n")
        random.seed(random.randint(0, 1000000))
        computer_shuffle = True

        players = []
        for config in player_configs:
            if config["type"] == "heuristic":
                players.append(HeuristicPlayer(config["name"]))
            elif config["type"] == "mcts":
                players.append(MCTSPlayer(config["name"],
                                           deals=config["deals"],
                                           rollouts=config["rollouts"],
                                           expl_rate=config["expl_rate"],
                                                 scoring=config["scoring"]))
            elif config["type"] == "nnmcts":
                players.append(NNMCTSPlayer(config["name"],
                                             config["model"],
                                             config["scaler"],
                                                   config["device"],
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
    except Exception as e:
        print(f"Simulation {idx} crashed with error: {e}")
        traceback.print_exc()
        return "CRASHED"

if __name__ == '__main__':
    total_games = 100
    print_every = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model for DeterminizedMLPMCTS
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "../models/hand_predict_mlp_25k_95epochs.pth")
    model_path = os.path.abspath(model_path)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

    # Load model weights
    model = HandPredictMLP(input_size=485, output_size=156)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Rebuild the scaler with loaded params
    scaler = MinMaxScaler()
    scaler.data_min_ = checkpoint['scaler_min']
    scaler.data_max_ = checkpoint['scaler_max']
    scaler.scale_ = checkpoint['scaler_scale']

    # For MinMaxScaler, set these attributes:
    scaler.min_ = np.zeros_like(scaler.scale_)
    scaler.data_range_ = scaler.data_max_ - scaler.data_min_
    scaler.n_samples_seen_ = np.array([1])  # dummy value, can be any positive integer

    print("Model and scaler loaded successfully.")

    expl_rate_list = [1.7]  # Exploration rates to test
    deals_list = [3, 5, 7, 10, 15, 20]  # Deals to test
    rollouts_list = [250]  # Rollouts to test

    for expl_rate in expl_rate_list:
        for deals in deals_list:
            for rollouts in rollouts_list:
                start_time = time.time()
                print(f"\n=== Testing deals={deals}, rollouts={rollouts}, exploration_rate={expl_rate} ===")

                player_configs = [
                    {"type": "heuristic", "name": "H1"},
                    {"type": "heuristic", "name": "H2"},
                    {"type": "heuristic", "name": "H3"},
                    {"type": "nnmcts", "name": "nnMCTS_test","model": model, "scaler": scaler, "device": device,
                     "deals": deals, "rollouts": rollouts, "expl_rate": expl_rate, "scoring": "win_rate"}]

                losses = {config["name"]: 0 for config in player_configs}

                with get_context("spawn").Pool(processes=cpu_count()) as pool:
                    run_func = partial(run_simulation, player_configs)
                    completed = 0
                    crashes = 0
                    timeouts = 0

                    results = [pool.apply_async(run_func, (i,)) for i in range(total_games)]

                    for i, r in enumerate(results):
                        try:
                            loser_name = r.get(timeout=600)  # Timeout after 600 seconds per game
                        except TimeoutError:
                            print(f"Simulation {i} timed out.")
                            timeouts += 1
                            continue
                        except Exception as e:
                            print(f"Simulation {i} raised an unexpected error: {e}")
                            crashes += 1
                            continue

                        if loser_name == "CRASHED":
                            crashes += 1
                            continue

                        losses[loser_name] += 1
                        completed += 1

                        if completed % print_every == 0:
                            print(f"{completed}/{total_games} games completed...", end='\r')

                # Results
                total_time = time.time() - start_time
                games_per_second = total_games / total_time

                print(f"\nResults for deals={deals}, rollouts={rollouts}, exploration_rate={expl_rate}:")
                print(f"Total games: {total_games}, Time: {total_time:.5f} s, Speed: {games_per_second:.5f} games/s")
                print(f"\n{crashes} simulations crashed and were skipped.")

                for player, loss_count in losses.items():
                    loserate = (loss_count / total_games) * 100
                    print(f'{player} losses: {loss_count} ({loserate:.2f}%)')

                # Memory cleanup
                del losses
                del run_func
                torch.cuda.empty_cache()
                gc.collect()