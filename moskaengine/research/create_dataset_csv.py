from moskaengine.players.random_player import RandomPlayer as Random
from moskaengine.players.determinized_mcts_player import DeterminizedMCTS
from moskaengine.game.engine import MoskaGame

import os
import numpy as np
import pandas as pd
import uuid
import time
from multiprocessing import Pool, cpu_count, Queue, Process


def run_simulation(seed):
    players = [Random('Random1'), Random('Random2'), Random('Random3'), Random('Random4')]
    computer_shuffle = True
    game = MoskaGame(players,
                     computer_shuffle,
                     save_vectors=True,
                     print_info=False
                     )
    while not game.is_end_state:
        game.next()

    print("Loser:", game.loser)

    return game.state_data, game.opponent_data

def writer(queue, max_batch_size_mb=50):
    save_folder_states = os.path.join("vectors/state")
    save_folder_opponents = os.path.join("vectors/opponent")
    os.makedirs(save_folder_states, exist_ok=True)
    os.makedirs(save_folder_opponents, exist_ok=True)

    state_batch = []
    opponent_batch = []
    batch_counter = 0
    job_uuid = uuid.uuid4()

    while True:
        item = queue.get()
        if item == "DONE":
            break
        state_data, opponent_data = item

        state_batch.append(np.array(state_data))
        opponent_batch.append(np.array(opponent_data))

        # Estimate total batch size in MB
        state_size = sum(arr.nbytes for arr in state_batch) / (1024 ** 2)
        opponent_size = sum(arr.nbytes for arr in opponent_batch) / (1024 ** 2)
        total_size = state_size + opponent_size

        if total_size >= max_batch_size_mb:
            # Stack numpy arrays
            state_stack = np.vstack(state_batch)
            opponent_stack = np.vstack(opponent_batch)

            # Convert to DataFrames
            state_df = pd.DataFrame(state_stack)
            opp_df = pd.DataFrame(opponent_stack)

            # Define file paths
            state_path = os.path.join(save_folder_states, f"states_batch_{job_uuid}_{batch_counter}.csv")
            opp_path = os.path.join(save_folder_opponents, f"opponents_batch_{job_uuid}_{batch_counter}.csv")

            # Write CSV
            state_df.to_csv(state_path, index=False, header=False)
            opp_df.to_csv(opp_path, index=False, header=False)

            batch_counter += 1
            state_batch.clear()
            opponent_batch.clear()

    # Save any remaining data when finishing
    if state_batch:
        state_stack = np.vstack(state_batch)
        opponent_stack = np.vstack(opponent_batch)

        state_df = pd.DataFrame(state_stack)
        opp_df = pd.DataFrame(opponent_stack)

        state_path = os.path.join(save_folder_states, f"states_batch_{job_uuid}_{batch_counter}.csv")
        opp_path = os.path.join(save_folder_opponents, f"opponents_batch_{job_uuid}_{batch_counter}.csv")

        state_df.to_csv(state_path, index=False, header=False)
        opp_df.to_csv(opp_path, index=False, header=False)


def main():
    total_games = 1
    print_every = 1

    queue = Queue(maxsize=cpu_count() * 2)

    writer_process = Process(target = writer, args=(queue,10_000_000))
    writer_process.start()
    completed = 0

    with Pool(processes=cpu_count()) as pool:
        for state_data, opponent_data in pool.imap_unordered(run_simulation, range(total_games)):
            queue.put((state_data, opponent_data))
            # print(len(state_data), len(opponent_data))
            if completed % print_every == 0:
                print(f"{completed}/{total_games} games completed...")
            completed += 1

    queue.put("DONE")
    writer_process.join()

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    avg = (end - start) / 2000
    print(f"Finished in {end - start:.2f} seconds")
    print(f"Average time per game: {avg:.8f} seconds")
