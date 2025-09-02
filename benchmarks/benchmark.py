import time
import os
import random
import traceback
import gc
import threading
import numpy as np
import torch
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

# Moskaengine imports
from research import HandPredictMLP
from moskaengine import MoskaGame, HeuristicPlayer, RandomPlayer, MCTSPlayer, NNMCTSPlayer

TIMEOUT_SECONDS = 60 * 30 # 30 minutes

benchmark_configs = [
    {
        "name": "MCTS_vs_2_Heuristic_vs_NN_MCTS",
        "players": [
            {"type": "mcts", "name": "MCTS_test", "deals": 10, "rollouts": 750, "expl_rate": 1.7, "scoring": "win_rate"},
            {"type": "heuristic", "name": "H1"},
            {"type": "heuristic", "name": "H2"},
            {"type": "nnmcts", "name": "NN_MCTS_test", "deals": 10, "rollouts": 750, "expl_rate": 1.7, "scoring": "win_rate"},
        ],
        "num_games": 10
    }
]

def run_simulation(player_configs, idx):
    try:
        print(f"Starting simulation {idx+1}...\n")
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

def run_with_timeout(func, timeout, *args, **kwargs):
    result_container = {}

    def target():
        try:
            result_container["result"] = func(*args, **kwargs)
        except Exception as e:
            result_container["error"] = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return None # Timed out
    if "error" in result_container:
        raise result_container["error"]

    return result_container.get("result")

if __name__ == '__main__':
    # TODO: Create a timeout for game if doesnt work.

    for config in benchmark_configs:
        total_games = config["num_games"]

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

        completed = 0
        crashes = 0

        for i in tqdm(range(total_games), desc=f"Running {config['name']}", unit="game"):
            try:
                loser_name = run_with_timeout(run_simulation, TIMEOUT_SECONDS, player_configs, i)

                if loser_name is None:
                    print(f"Game {i+1} timed out after {TIMEOUT_SECONDS/60} minutes, skipping...")
                    continue

                if loser_name == "CRASHED":
                    crashes += 1
                    continue

                losses[loser_name] += 1
                completed += 1

            except Exception as e:
                print(f"Unexpected error in simulation {i}: {e}")
                traceback.print_exc()
                crashes += 1
                continue

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
        results_file = f"results_{config['name'].replace(' ', '_').lower()}_{random.randint(0,1000)}.txt"
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
        torch.cuda.empty_cache()
        gc.collect()