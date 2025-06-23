from moskaengine.players.abstract_player import AbstractPlayer
from moskaengine.mcts.mcts import MCTS
from moskaengine.game.deck import Card
import random
from collections import deque
from moskaengine.utils.game_utils import check_unique_game_state


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
        """Determinize the game state by randomly assigning unknown cards to players."""

        # Make each card in our hand public
        for hand_card in game_state.player_to_play.hand:
            hand_card.is_private = False
            hand_card.is_public = True

            # Update matching card in card_collection
            for collection_card in game_state.card_collection:
                if collection_card.suit == hand_card.suit and collection_card.value == hand_card.value:
                    collection_card.is_unknown = False
                    collection_card.is_private = False
                    collection_card.is_public = True
                    break

        # Get unknown cards
        unknown_cards = list(game_state.get_non_public_cards())
        if not unknown_cards:
            return game_state  # Nothing to assign

        # Identify opponents and their needed card counts
        opponents = [player for player in game_state.players if player != game_state.player_to_play]
        original_hand_sizes = {player: len(player.hand) for player in opponents}

        # Remove any private cards from opponent hands (if any)
        for player in opponents:
            player.hand = [card for card in player.hand if card.is_public]

        # Remove unknown cards from the deck safely
        remaining_deck_cards = [card for card in game_state.deck.cards if not card.is_unknown]
        cards_removed_from_deck = len(game_state.deck.cards) - len(remaining_deck_cards)
        game_state.deck.cards = deque(remaining_deck_cards)

        unknown_cards_per_player = {
            player: original_hand_sizes[player] - len(player.hand)
            for player in opponents
        }

        # Shuffle unknown cards for random assignment
        random.shuffle(unknown_cards)

        # Assign cards randomly to opponents
        for player in opponents:
            num_needed = unknown_cards_per_player[player]
            for _ in range(num_needed):
                if not unknown_cards:
                    break
                card_to_assign = unknown_cards.pop(0)
                new_card = Card()
                new_card.from_card(card_to_assign)
                new_card.is_public = True
                new_card.is_private = False
                new_card.is_unknown = False
                player.hand.append(new_card)

        # All of the remaining unknown cards go to the deck
        for card_to_assign in unknown_cards.copy():
            new_card = unknown_cards.pop(0)
            new_card.from_card(card_to_assign)
            new_card.is_public = True
            new_card.is_private = False
            new_card.is_unknown = False
            # Add to the beginning of the deck
            game_state.deck.cards.appendleft(new_card)

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

            # Check uniqueness of the determinized state
            check_unique_game_state(copied)

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
            action_to_play, (W, N) = max(valid_total_ratings.items(), key=lambda x: x[1][0] / (x[1][1] + 1e-10))
            if game_state.print_info:
                print(f'{self} expects to not lose with {W/N*100:.2f}%')
        else:
            raise BaseException('Scoring type not implemented.')

        # Not needed
        # self.mcts = search_tree

        # assert action_to_play in game_state.allowed_plays(), f"Action {action_to_play} not allowed in {game_state.allowed_plays()}"

        # Make own cards private again
        for card in game_state.player_to_play.hand:
            if card.is_public:
                card.is_private = True
                card.is_public = False

        return action_to_play