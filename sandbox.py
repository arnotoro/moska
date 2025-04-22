import random
from moskaengine.game.game import MoskaGame
from moskaengine.players.human_player import Human
from moskaengine.players.random_player import RandomPlayer as Random

siemen = 618027
print(siemen)
random.seed(siemen)

random.seed(random.randint(0, 1000000))


# print(siemen)

# Note the main attacker should be specified
# The players can be one of ISMCTS, ISMCTSFPV, DeterminizedMCTS, Random, Human
# players = [Human('Player1'), Human('Player2')]
players = [Random('Player1'), Random('Player2')]

# If the computer must shuffle the deck of cards instead the player in real-life
# computer_shuffle = False
computer_shuffle = True

game = MoskaGame(players, computer_shuffle, main_attacker='Player1')
while not game.is_end_state:
    game.next()

print()
print(f'Game is lost by {game.loser}')