from moskaengine.players.abstract_player import AbstractPlayer
from moskaengine.utils.card_utils import choose_random_action, basic_repr_game


class RandomPlayer(AbstractPlayer):
    def make_copy(self):
        new = RandomPlayer(self.name)
        new.hand = self.hand.copy()
        return new

    def choose_action(self, game_state):
        # Choose a random action from the allowed actions
        self.make_cards_known(game_state)
        # print(f"Random players hand: {self.hand}")

        # Get the allowed actions
        allowed_actions = game_state.allowed_plays()

        random_action = choose_random_action(allowed_actions)
        return random_action