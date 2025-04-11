from moskaengine.players.random_player import RandomPlayer as Random
from moskaengine.game.game import MoskaGame
import random
import time
from multiprocessing import Pool, cpu_count

def run_simulation(seed):
    players = [Random('Random1'), Random('Random2'), Random('Random3'), Random('Random4')]
    computer_shuffle = True
    game = MoskaGame(players, computer_shuffle, main_attacker=random.choice(['Random1', 'Random2', 'Random3', 'Random4']), print_info=False)
    while not game.is_end_state:
        game.next()
    return str(game.loser).replace('Player ', '')

if __name__ == '__main__':
    total_games = 1_000_000
    print_every = 10000
    start_time = time.time()

    losses = {name: 0 for name in ['Random1', 'Random2', 'Random3', 'Random4']}  # or however many players

    with Pool(processes=cpu_count()) as pool:
        completed = 0
        for loser_name in pool.imap_unordered(run_simulation, range(total_games)):
            losses[loser_name] += 1
            completed += 1
            if completed % print_every == 0:
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
