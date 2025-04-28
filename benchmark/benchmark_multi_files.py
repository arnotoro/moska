from moskaengine.players.random_player import RandomPlayer as Random
from moskaengine.game.game import MoskaGame
from moskaengine.utils.game_utils import save_game_vector, save_game_state_vector_batch, save_opponent_vector_batch

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import uuid
import time
from multiprocessing import Pool, cpu_count, Queue, Process


def run_simulation(seed):
    players = [Random('Random1'), Random('Random2'), Random('Random3'), Random('Random4')]
    computer_shuffle = True
    game = MoskaGame(players,
                     computer_shuffle,
                     save_vectors=True,
                     )
    while not game.is_end_state:
        game.next()
    return game.state_data, game.opponent_data

def writer(queue, batch_size=10_000_000):
    save_folder_states = os.path.join("../vectors/state")
    save_folder_opponents = os.path.join("../vectors/opponent")
    os.makedirs(save_folder_states, exist_ok=True)
    os.makedirs(save_folder_opponents, exist_ok=True)

    state_batch = []
    opponent_batch = []
    batch_counter = 0

    while True:
        item = queue.get()
        if item == "DONE":
            break
        state_data, opponent_data = item
        state_batch.extend(state_data)
        opponent_batch.extend(opponent_data)

        if len(state_batch) >= batch_size:
            state_df = pd.DataFrame(state_batch)
            opp_df = pd.DataFrame(opponent_batch)

            state_path = os.path.join(save_folder_states, f"states_batch_{uuid.uuid4()}_{batch_counter}.parquet")
            opp_path = os.path.join(save_folder_opponents, f"opponents_batch_{uuid.uuid4()}_{batch_counter}.parquet")

            state_df.to_parquet(state_path, engine='pyarrow')
            opp_df.to_parquet(opp_path, engine='pyarrow')

            batch_counter += 1
            state_batch.clear()
            opponent_batch.clear()

    # Save any remaining data when finishing
    if state_batch:
        state_df = pd.DataFrame(state_batch)
        opp_df = pd.DataFrame(opponent_batch)

        state_path = os.path.join(save_folder_states, f"states_batch_{uuid.uuid4()}_{batch_counter}.parquet")
        opp_path = os.path.join(save_folder_opponents, f"opponents_batch_{uuid.uuid4()}_{batch_counter}.parquet")

        state_df.to_parquet(state_path, engine='pyarrow')
        opp_df.to_parquet(opp_path, engine='pyarrow')

def main():
    total_games = 2000
    print_every = 100

    queue = Queue()

    writer_process = Process(target = writer, args=(queue,10_000_000))
    writer_process.start()
    completed = 0

    with Pool(processes=cpu_count()) as pool:
        for state_data, opponent_data in pool.imap_unordered(run_simulation, range(total_games)):
            queue.put((state_data, opponent_data))

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
    print(f"Average time per game: {avg:.2f} seconds")
