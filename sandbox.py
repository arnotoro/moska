import random
import time


from moskaengine.game.engine import MoskaGame
from moskaengine.players.human_player import Human
from moskaengine.players.random_player import RandomPlayer as Random
from moskaengine.players.determinized_mcts_player import DeterminizedMCTS
from moskaengine.players.heuristic_player import HeuristicPlayer as Heuristic

siemen = random.randint(0, 1000000)
print(f"Used seed: {siemen}")
random.seed(2539)

# random.seed(303384)

# print(siemen)

# Note the main attacker should be specified
# The players can be one of ISMCTS, ISMCTSFPV, DeterminizedMCTS, Random, Human
# players = [Random('Player1'), Random('Player2'), Random('Player3'), DeterminizedMCTS('MCTS', deals=5, rollouts=200, expl_rate=0.7, scoring="win_rate")]
# players = [Random('Random'), DeterminizedMCTS('MCTS', deals=5, rollouts=200, expl_rate=0.7, scoring="win_rate")]
players = [Heuristic('Heuristic'), Heuristic('Random1')] #, Heuristic('Random2'), Heuristic('Random3')]

# If the computer must shuffle the deck of cards instead the player in real-life
# computer_shuffle = False
computer_shuffle = True

start = time.time()
game = MoskaGame(players, computer_shuffle, perfect_info=False, save_vectors=True, print_info=True)
while not game.is_end_state:
    game.next()

end = time.time()
print(f"Game took {end - start:.2f} seconds")
# print(game.state_data, game.opponent_data)
print(f'Game is lost by {game.loser}')