from moskaengine.players.determinized_mcts_player import DeterminizedMCTS
from moskaengine.players.random_player import RandomPlayer as Random
from moskaengine.players.determinized_mcts_player import DeterminizedMCTS
from moskaengine.players.heuristic_player import HeuristicPlayer as Heuristic

from moskaengine.game.engine import MoskaGame
import random
import time

players = [Heuristic('Heuristic'), DeterminizedMCTS('MCTS', deals=5, rollouts=100, expl_rate=0.7, scoring="win_rate")]
# players = [Random('Random1'), Random('Random2'), Heuristic('Heuristic3'), Heuristic('Heuristic4')]
computer_shuffle = True

# Track losses for each player
losses = {player.name: 0 for player in players}  # or however many players
total_games = 5

# Start timing
start_time = time.time()

for i in range(total_games):
    random.seed(random.randint(0, 1000000))
    game = MoskaGame(players, computer_shuffle, print_info=False)
    while not game.is_end_state:
        game.next()

    # Record the loser directly
    print(f'Game {i + 1} is lost by {game.loser} with hand {game.loser.hand}')
    loser_name = str(game.loser).replace('Player ', '')  # Clean up the name if needed
    losses[loser_name] += 1

    # Optional: Print progress
    if i % 1 == 0 and i > 0:
        elapsed_time = time.time() - start_time
        games_per_second = i / elapsed_time
        print(f'Completed {i} games. Speed: {games_per_second:.2f} games/second')

# Calculate total time and speed
total_time = time.time() - start_time
games_per_second = total_games / total_time

# Print final results
print()
print(f'Total games played: {total_games}')
print(f'Total simulation time: {total_time:.2f} seconds')
print(f'Average speed: {games_per_second:.2f} games/second')
for name, loss in losses.items():
    print(f'{name} losses: {loss} ({(loss / total_games) * 100:.2f}%)')