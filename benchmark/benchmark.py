from moskaengine.players.random_player import RandomPlayer as Random
from moskaengine.game.game import MoskaGame
import random
import time

players = [Random('Random1'), Random('Random2')]
computer_shuffle = True

# Track losses for each player
losses = {'Random1': 0, 'Random2': 0}
total_games = 1000

# Start timing
start_time = time.time()

# print(random.choice(['Random1', 'Random2']))

for i in range(total_games):
    random.seed(random.randint(0, 1000000))
    game = MoskaGame(players, computer_shuffle, main_attacker='Random1', print_info=False)
    while not game.is_end_state:
        game.next()

    # Record the loser directly
    loser_name = str(game.loser).replace('Player ', '')  # Clean up the name if needed
    losses[loser_name] += 1

    # Optional: Print progress
    if i % 100 == 0 and i > 0:
        elapsed_time = time.time() - start_time
        games_per_second = i / elapsed_time
        print(f'Completed {i} games. Speed: {games_per_second:.2f} games/second')

# Calculate total time and speed
total_time = time.time() - start_time
games_per_second = total_games / total_time

# Calculate winrates
winrate_random1 = ((total_games - losses['Random1']) / total_games) * 100
winrate_random2 = ((total_games - losses['Random2']) / total_games) * 100

print()
print(f'Total games played: {total_games}')
print(f'Total simulation time: {total_time:.2f} seconds')
print(f'Average speed: {games_per_second:.2f} games/second')
print(f'Random1 wins: {total_games - losses["Random1"]} ({winrate_random1:.2f}%)')
print(f'Random2 wins: {total_games - losses["Random2"]} ({winrate_random2:.2f}%)')