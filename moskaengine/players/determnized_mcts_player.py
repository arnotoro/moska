from moskaengine.players.abstract_player import AbstractPlayer
from moskaengine.mcts.mcts import MCTS
import random

class DeterminizedMCTS(AbstractPlayer):
    """


    """
    def __init__(self, name, deals=10, rollouts=100, expl_rate=0.7, scoring="win_rate"):
        super().__init__(name)
        self.name = name
        self.hand = []
        self.mcts = MCTS()
        self.deals = deals
        self.scoring = scoring
        self.rollouts = rollouts
        self.expl_rate = expl_rate


    def make_copy(self):
        new = DeterminizedMCTS(self.name)
        new.hand = self.hand.copy()
        return new

    def randomly_determinize(self, game_state):
        """Randomly determinize the game state to a deterministic state."""

        # Make each card in our hand public
        for card in game_state.player_to_play.hand:
            card.is_private = False
            card.is_public = True

        # Unknown cards are shuffled
        unknown = list(game_state.get_non_public_cards())
        random.shuffle(unknown)

        # Define the unknown cards in the card collection as random cards
        for card in game_state.card_collection:
            if card.is_unknown:
                suit, value = unknown.pop(0)
                card.from_suit_value(suit, value)
                card.is_public = True
            elif card.is_private:
                suit, value = unknown.pop(0)
                card.is_private = False
                card.is_unknown = True
                card.from_suit_value(suit, value)
                card.is_public = True

        return game_state

    def choose_action(self, game_state):
        """Choose an action from all allowed actions by using a determinized MCTS.
         The hand of the player must be known to determine which plays are allowed."""

        self.make_cards_known(game_state)

        # Check if only one action is allowed to possibly save time
        allowed = game_state.allowed_plays()
        if len(allowed) == 1:
            return allowed[0]

        copied_state = game_state.clone_for_rollout()

        for card in copied_state.card_collection:
            if card.is_private and card not in self.hand:
                card.reset(card.suit, card.value)

        total_ratings = {}
        for deal in range(self.deals):
            copied = copied_state.clone_for_rollout()

            # Determinize the game state
            copied = self.randomly_determinize(copied)
            # TODO: This messes up the allowed actions

            # Search tree from previous iterations
            search_tree = game_state.player_to_play.mcts

            # Rollouts
            action_ratings = search_tree.do_rollout(copied, self.rollouts, self.expl_rate)
            for action, (W, N) in action_ratings.items():
                if action not in total_ratings:
                    total_ratings[action] = (W, N)
                else:
                    old_W, old_N = total_ratings[action]
                    # Update the total ratings
                    total_ratings[action] = (old_W + W, old_N + N)

        # Get allowed plays from the real (non-determinized) game state
        allowed_actions = set(game_state.allowed_plays())
        # Filter total_ratings to only include allowed actions
        valid_total_ratings = {action: (W, N) for action, (W, N) in total_ratings.items() if
                               action in allowed_actions}

        if not valid_total_ratings:
            raise Exception("No valid actions found in MCTS results — possible determinization issue.")

        # We choose the action to perform based on the total statistics
        if self.scoring == "visit_count":
            action_to_play, (W, N) = max(valid_total_ratings.items(), key=lambda x: x[1][1])
            if game_state.print_info:
                print(f'{self} expects to not lose with {N} visits')
        elif self.scoring == "win_rate":
            action_to_play, (W, N) = max(valid_total_ratings.items(), key=lambda x: x[1][0] / x[1][1])
            if game_state.print_info:
                print(f'{self} expects to not lose with {W/N*100:.2f}%')
        else:
            raise BaseException('Scoring type not implemented.')

        self.mcts = search_tree

        # assert action_to_play in game_state.allowed_plays(), f"Action {action_to_play} not allowed in {game_state.allowed_plays()}"

        return action_to_play


