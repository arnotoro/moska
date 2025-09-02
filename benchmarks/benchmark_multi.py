# General imports
import os
import time
import traceback
from multiprocessing import Pool, cpu_count, get_context
from functools import partial
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import random
import gc

# Moskaengine imports
from research import HandPredictMLP
from moskaengine import MoskaGame, HeuristicPlayer, RandomPlayer, MCTSPlayer, NNMCTSPlayer

# Benchmark configurations
# Each configuration contains player types, names, number of games, and chunk size for parallel processing

benchmark_configs = [
    {
        "name": "MCTS_vs_2_Heuristic_vs_NN_MCTS",
        "players": [
            {"type": "heuristic", "name": "H1"},
            {"type": "mcts", "name": "MCTS_test", "deals": 10, "rollouts": 500, "expl_rate": 1.7, "scoring": "win_rate"},
            {"type": "heuristic", "name": "H2"},
            {"type": "nnmcts", "name": "NN_MCTS_test", "deals": 10, "rollouts": 500, "expl_rate": 1.7, "scoring": "win_rate"},
        ],
        "num_games": 100
    }
]


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
            elif config["type"] == "random":
                players.append(RandomPlayer(config["name"]))
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

def load_nnmcts_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model for DeterminizedMLPMCTS
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "../models/hand_predict_mlp_50k_83epochs.pth")
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
    return model, scaler, device


if __name__ == '__main__':
    for config in benchmark_configs:
        total_games = config["num_games"]
        print_every = 10

        print(f"\n=== Benchmarking {config['name']} ===")

        player_configs = config["players"]

        if any(p["type"] == "nnmcts" for p in player_configs):
            model, scaler, device = load_nnmcts_model()
            for p in player_configs:
                if p["type"] == "nnmcts":
                    p["model"] = model
                    p["scaler"] = scaler
                    p["device"] = device

        losses = {config["name"]: 0 for config in player_configs}

        start_time = time.time()


        run_func = partial(run_simulation, player_configs)

        with get_context("spawn").Pool(processes=cpu_count()) as pool:
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

        print(f"\nResults for {config['name']}:")
        print(f"Total games: {total_games}, Time: {total_time:.5f} s, Speed: {games_per_second:.5f} games/s")
        print(f"\n{crashes} simulations crashed and were skipped.")

        for player, loss_count in losses.items():
            loserate = (loss_count / total_games) * 100
            print(f'{player} losses: {loss_count} ({loserate:.2f}%)')

        # Save results to a file
        results_file = f"results_{config['name'].replace(' ', '_').lower()}.txt"
        with open(results_file, 'w') as f:
            f.write(f"Results for {config['name']}:\n")
            f.write(f"Total games: {total_games}, Time: {total_time:.5f} s, Speed: {games_per_second:.5f} games/s\n")
            f.write(f"\n{crashes} simulations crashed and were skipped.\n\n")

            for player, loss_count in losses.items():
                loserate = (loss_count / total_games) * 100
                f.write(f'{player} losses: {loss_count} ({loserate:.2f}%)\n')

        print(f"Results saved to {results_file}")

        # Memory cleanup
        del losses
        del run_func
        torch.cuda.empty_cache()
        gc.collect()