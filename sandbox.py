import random
from moskaengine.game.game import MoskaGame
from moskaengine.players.human_player import Human
from moskaengine.players.random_player import RandomPlayer as Random

siemen = random.randint(0, 1000000)
print(f"Used seed: {siemen}")
random.seed(siemen)

# random.seed(303384)


# print(siemen)

# Note the main attacker should be specified
# The players can be one of ISMCTS, ISMCTSFPV, DeterminizedMCTS, Random, Human
# players = [Human('Player1'), Random('Player2'), Random('Player3'), Random('Player4')]
players = [Random('Player1'), Random('Player2')]

# If the computer must shuffle the deck of cards instead the player in real-life
# computer_shuffle = False
computer_shuffle = True

game = MoskaGame(players, computer_shuffle, perfect_info=False, save_vectors=True, print_info=True)
while not game.is_end_state:
    game.next()

# print(game.state_data, game.opponent_data)
print(f'Game is lost by {game.loser}')