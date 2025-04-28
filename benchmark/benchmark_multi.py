from moskaengine.players.random_player import RandomPlayer as Random
from moskaengine.game.game import MoskaGame
from moskaengine.utils.game_utils import save_game_vector, save_game_state_vector_batch, save_opponent_vector_batch
import os
import time
import uuid
import csv
from multiprocessing import Pool, cpu_count, Queue, Process

def run_simulation(seed):
    players = [Random('Random1'), Random('Random2'), Random('Random3'), Random('Random4')]
    computer_shuffle = True
    game = MoskaGame(players,
                     computer_shuffle,
                     save_vectors=True,
                     state_folder="multi_test_100"

                     )
    while not game.is_end_state:
        game.next()
    return str(game.loser).replace('Player ', ''), game.state_data, game.opponent_data

def writer(queue):
    save_folder_states = os.path.join(f"../vectors/states")
    save_folder_opponents = os.path.join(f"../vectors/opponents")
    os.makedirs(save_folder_states, exist_ok=True)
    os.makedirs(save_folder_opponents, exist_ok=True)

    state_path = os.path.join(save_folder_states, f"states_{uuid.uuid4()}.csv")
    opponent_path = os.path.join(save_folder_opponents, f"opponents_{uuid.uuid4()}.csv")

    with open(state_path, 'a', newline='') as state_file, open(opponent_path, 'a', newline='') as opp_file:
        state_writer = csv.writer(state_file)
        opp_writer = csv.writer(opp_file)

        while True:
            item = queue.get()
            if item == "DONE":
                break
            state_data, opponent_data = item
            state_writer.writerows(state_data)
            opp_writer.writerows(opponent_data)

if __name__ == '__main__':
    total_games = 2000
    print_every = 100
    start_time = time.time()

    losses = {name: 0 for name in ['Random1', 'Random2', 'Random3', 'Random4']}  # or however many players
    results_buffer_state = []
    results_buffer_opponent = []
    batch_number = 1

    # Get the directory for saving the game vectors


    with Pool(processes=cpu_count()) as pool:
        completed = 0
        for loser_name, state_data, opponent_data in pool.imap_unordered(run_simulation, range(total_games)):
            losses[loser_name] += 1
            # print(state_data)
            results_buffer_state.extend(state_data)
            # print(results_buffer_state)
            results_buffer_opponent.extend(opponent_data)
            completed += 1

            if completed % print_every == 0:
                save_game_state_vector_batch(results_buffer_state, batch_number)
                save_opponent_vector_batch(results_buffer_opponent, batch_number)
                results_buffer_state.clear()
                results_buffer_opponent.clear()
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
