from moskaengine.players.abstract_player import AbstractPlayer
from moskaengine.utils.card_utils import choose_random_action

class RandomPlayer(AbstractPlayer):
    def make_copy(self):
        new = RandomPlayer(self.name)
        new.hand = self.hand.copy()
        return new

    def choose_action(self, game_state):
        # Choose a random action from the allowed actions
        self.make_cards_known(game_state)

        # Get the allowed actions
        allowed_actions = game_state.allowed_plays()

        return choose_random_action(allowed_actions)