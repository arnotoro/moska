# Global imports
import time
import random
from multiprocessing import Pool, cpu_count, Queue, Process
from functools import partial

# Local imports
from moskaengine.game.engine import MoskaGame
# Players
from moskaengine.players.determinized_mcts_player import DeterminizedMCTS
from moskaengine.players.heuristic_player import HeuristicPlayer as Heuristic
from moskaengine.players.random_player import RandomPlayer as Random

def run_simulation(players_list, random_seed):
    random.seed(random.randint(0, 1000000))
    computer_shuffle = True
    game = MoskaGame(players_list, computer_shuffle, save_vectors=False, print_info=False)
    while not game.is_end_state:
        game.next()
    return str(game.loser).replace('Player ', ''), game.state_data, game.opponent_data

if __name__ == '__main__':
    total_games = 100
    print_every = 10
    start_time = time.time()

    players = [Random('Random1'), Random('Random2'), Random('Random3'), DeterminizedMCTS('MCTS_reference', deals=3, rollouts=250, expl_rate=0.7, scoring="win_rate")]
    # players = [Heuristic('Heuristic1'), Heuristic('Heuristic2'), Heuristic('Heuristic3'), DeterminizedMCTS('MCTS_reference', deals=1, rollouts=100, expl_rate=0.7, scoring="win_rate")]
    # players = [DeterminizedMCTS('MCTS_reference1', deals=3, rollouts=100, expl_rate=1, scoring="win_rate"),
    #            DeterminizedMCTS('MCTS_reference2', deals=3, rollouts=100, expl_rate=1, scoring="win_rate"),
    #            DeterminizedMCTS('MCTS_reference3', deals=3, rollouts=100, expl_rate=1, scoring="win_rate"),
    #            DeterminizedMCTS('MCTS_reference4', deals=3, rollouts=100, expl_rate=1, scoring="win_rate")]
    losses = {player.name: 0 for player in players}

    with Pool(processes=cpu_count()) as pool:
        completed = 0
        run_func = partial(run_simulation, players)

        for loser_name, state_data, opponent_data in pool.imap_unordered(run_func, range(total_games)):
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

    # Print loses for each player
    for player, loss_count in losses.items():
        lost_rate = (loss_count / total_games) * 100
        print(f'{player} losses: {loss_count} ({lost_rate:.2f}%)')
