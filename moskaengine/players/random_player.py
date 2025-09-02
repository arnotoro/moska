from .abstract_player import AbstractPlayer
from ..utils import choose_random_action

class RandomPlayer(AbstractPlayer):
    def make_copy(self):
        new = RandomPlayer(self.name)
        new.hand = self.hand.copy()
        return new

    def choose_action(self, game_state):
        # Choose a random action from the allowed actions
        self.make_cards_known(game_state)
        # print(f"{self.name} players hand: {self.hand}")

        # Get the allowed actions
        allowed_actions = game_state.allowed_plays()

        random_action = choose_random_action(allowed_actions)
        return random_action