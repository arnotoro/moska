import os
import time
import datetime
import uuid
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from multiprocessing import Pool, cpu_count, Queue, Process

# Moskaengine imports
from moskaengine import MoskaGame, HeuristicPlayer, RandomPlayer, MCTSPlayer

def run_simulation(seed):
    players = [HeuristicPlayer('Heuristic1'), HeuristicPlayer('Heuristic2'), HeuristicPlayer('Heuristic3'), HeuristicPlayer('Heuristic4')]
    computer_shuffle = True
    game = MoskaGame(players,
                     computer_shuffle,
                     save_vectors=True,
                     print_info=False
                     )
    while not game.is_end_state:
        game.next()

    return game.state_data, game.opponent_data

def writer(queue, max_batch_size_mb=50, total_games=1):
    # Add timestamp to ensure uniqueness
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    save_folder_states = os.path.join(f"vectors_{total_games}/state")
    save_folder_opponents = os.path.join(f"vectors_{total_games}/opponent")

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

            state_path = os.path.join(save_folder_states, f"states_batch_{job_uuid}_{batch_counter}.parquet")
            opp_path = os.path.join(save_folder_opponents, f"opponents_batch_{job_uuid}_{batch_counter}.parquet")

            # Convert numpy array to pyarrow Table
            state_table = pa.Table.from_arrays(
                [pa.array(state_stack[:, i]) for i in range(state_stack.shape[1])],
                names=[f"col_{i}" for i in range(state_stack.shape[1])]
            )

            # Write parquet directly
            pq.write_table(state_table, state_path, compression='snappy')

            opp_table = pa.Table.from_arrays(
                [pa.array(opponent_stack[:, i]) for i in range(opponent_stack.shape[1])],
                names=[f"col_{i}" for i in range(opponent_stack.shape[1])]
            )

            pq.write_table(opp_table, opp_path, compression='snappy')

            batch_counter += 1
            state_batch.clear()
            opponent_batch.clear()

    # Save any remaining data when finishing
    if state_batch:
        state_stack = np.vstack(state_batch)
        opponent_stack = np.vstack(opponent_batch)

        state_path = os.path.join(save_folder_states, f"states_batch_{job_uuid}_{batch_counter}.parquet")
        opp_path = os.path.join(save_folder_opponents, f"opponents_batch_{job_uuid}_{batch_counter}.parquet")

        # Convert numpy array to pyarrow Table
        state_table = pa.Table.from_arrays(
            [pa.array(state_stack[:, i]) for i in range(state_stack.shape[1])],
            names=[f"col_{i}" for i in range(state_stack.shape[1])]
        )

        # Write parquet directly
        pq.write_table(state_table, state_path, compression='snappy')

        opp_table = pa.Table.from_arrays(
            [pa.array(opponent_stack[:, i]) for i in range(opponent_stack.shape[1])],
            names=[f"col_{i}" for i in range(opponent_stack.shape[1])]
        )

        pq.write_table(opp_table, opp_path, compression='snappy')

if __name__ == '__main__':
    total_games = 50
    print_every = 10

    queue = Queue(maxsize=cpu_count() * 2)

    writer_process = Process(target = writer, args=(queue, 10_000_000, total_games))
    writer_process.start()
    completed = 0

    start = time.time()
    with Pool(processes=cpu_count()) as pool:
        for state_data, opponent_data in pool.imap_unordered(run_simulation, range(total_games)):
            queue.put((state_data, opponent_data))
            # print(len(state_data), len(opponent_data))
            if completed % print_every == 0:
                print(f"{completed}/{total_games} games completed...")
            completed += 1

    queue.put("DONE")
    writer_process.join()
    end = time.time()
    avg = (end - start) / total_games
    print(f"Finished in {end - start:.2f} seconds")
    print(f"Average time per game: {avg:.8f} seconds")
