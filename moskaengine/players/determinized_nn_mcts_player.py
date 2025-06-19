from moskaengine.players.abstract_player import AbstractPlayer
from moskaengine.mcts.mcts import MCTS
from moskaengine.utils.game_utils import state_as_vector
from moskaengine.research.model_training.OLD_train_model import CardPredictorMLP
from moskaengine.utils.game_utils import check_unique_game_state
from moskaengine.game.deck import Card, StandardDeck
import random
import torch
import numpy as np
from collections import deque

class DeterminizedMLPMCTS(AbstractPlayer):
    """


    """
    def __init__(self, name, model, device, deals=1, rollouts=100, expl_rate=0.7, scoring="win_rate", scaler = None):
        super().__init__(name)
        self.name = name
        self.hand = []
        self.mcts = MCTS()
        self.deals = deals
        self.scoring = scoring
        self.rollouts = rollouts
        self.expl_rate = expl_rate
        self.device = device
        self.model = model.to(self.device)
        self.scaler = scaler # Placeholder for scaler, if used in the future


    def make_copy(self):
        new = DeterminizedMLPMCTS(self.name, self.model, self.device)
        new.hand = self.hand.copy()
        return new

    def determinize_with_model(self, game_state):
        """Determinize the game state using model predictions."""
        # Make each card in our hand public and in the card collection
        for hand_card in game_state.player_to_play.hand:
            hand_card.is_private = False
            hand_card.is_public = True

            # Find and update the matching card in the card_collection
            for collection_card in game_state.card_collection:
                if collection_card.suit == hand_card.suit and collection_card.value == hand_card.value:
                    # Make it public in the collection too
                    collection_card.is_unknown = False
                    collection_card.is_private = False
                    collection_card.is_public = True
                    break

        # Get unknown cards and their tuples (suit, value pairs)
        unknown_cards = list(game_state.get_non_public_cards())
        unknown_tuples = list(game_state.get_non_public_cards_tuples())

        # If there are no unknown cards, return the game state as is
        if not unknown_cards:
            return game_state

        # Encode the current game state into input vector
        input_state_vector, _ = state_as_vector(game_state)

        # If a scaler is provided, scale the input state vector
        if self.scaler is not None:
            # Transform to numpy array
            input_state_vector = np.array(input_state_vector, dtype=np.float32)
            # Find non-binary features
            non_binary_columns = np.where((input_state_vector != 0) & (input_state_vector != 1))[0]
            # Remove the first column (deck size) from scaling
            non_binary_columns = non_binary_columns[non_binary_columns != 0]  # Exclude the first column
            # Scaler is applied to non-binary features e.g. anything other than 0 or 1 values
            values_to_scale = input_state_vector[non_binary_columns].reshape(-1, 1)  # Reshape for scaler

            # Fit and transform the non-binary features
            scaled_values = self.scaler.fit_transform(values_to_scale)

            # Apply a fixed scaling to the first column (deck size)
            normalized_deck = len(game_state.deck.cards) / 52.0 # Normalize deck size to [0, 1]
            input_state_vector[0] = normalized_deck
            # Flatten back to original shape
            input_state_vector[non_binary_columns] = scaled_values.flatten()
            # input_state_vector[non_binary_columns] = self.scaler.fit_transform(input_state_vector[non_binary_columns])

        # Convert the input state vector to a tensor
        input_state_tensor = torch.tensor(input_state_vector, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Use the model to predict the probabilities of each card being in each opponent's hand
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_state_tensor)
            # Apply sigmoid to get probabilities of multi-label classification
            probs = torch.sigmoid(logits).squeeze()

        # Reshape probabilities to match the number of opponents
        num_opponents = len(game_state.players) - 1
        probs = probs.view(num_opponents, 52)

        # Get the number of hidden cards for each opponent
        opponents = [player for player in game_state.players if player != game_state.player_to_play]
        original_hand_sizes = {player: len(player.hand) for player in opponents}

        # ### DEBUG ###
        # # Correct cards
        # for i, player in enumerate(opponents):
        #     print(f"Opponent {i} actual cards: {player.hand}")
        #     print()  # New line after each opponent's cards
        # ##################


        # Ensure each player's hand is cleared of private cards
        for player in opponents:
            player.hand = [card for card in player.hand if card.is_public]

        # Remove unknown cards from the deck safely
        remaining_deck_cards = [card for card in game_state.deck.cards if not card.is_unknown]
        cards_removed_from_deck = len(game_state.deck.cards) - len(remaining_deck_cards)
        game_state.deck.cards = deque(remaining_deck_cards)

        unknown_cards_per_player = {p: original_hand_sizes[p] - len(p.hand) for p in opponents}

        # Create a reference deck to map card indices to actual cards
        reference_deck = StandardDeck(shuffle=False, perfect_info=True)
        card_mapping = {}

        for idx, card in enumerate(reference_deck.cards):
            card_mapping[idx] = card

        # Assign cards to players based on probabilities
        assigned_cards = set()

        # For each player, assign cards with the highest probabilities
        for idx, player in enumerate(opponents):
            if idx >= len(probs):  # Safety check
                continue

            num_cards_needed = unknown_cards_per_player[player]
            if num_cards_needed <= 0:
                continue

            # Create pairs of (card_idx, probability)
            player_probs = probs[idx]
            card_prob_pairs = [(i, prob) for i, prob in enumerate(player_probs) if i in card_mapping]

            # Sort by probability (highest first)
            card_prob_pairs.sort(key=lambda x: x[1], reverse=True)

            # Assign cards to this player based on the probabilities
            cards_assigned_to_player = 0

            while cards_assigned_to_player < num_cards_needed and card_prob_pairs:
                card_idx, _ = card_prob_pairs.pop(0)
                card_to_assign = card_mapping[card_idx]

                # If the card is already assigned, go to the next one
                if card_to_assign in assigned_cards:
                    # print(f"Card {card_to_assign} already assigned, skipping.")
                    continue

                elif card_to_assign in unknown_cards:
                    # print(f"Assigning card {card_to_assign} to player {player.name}")
                    new_card = Card()
                    new_card.from_card(card_to_assign)
                    new_card.is_public = True
                    new_card.is_private = False
                    new_card.is_unknown = False

                    player.hand.append(new_card)
                    assigned_cards.add(card_to_assign)
                    # Remove the card from the unknown cards list
                    unknown_cards.remove(card_to_assign)
                    cards_assigned_to_player += 1
                else:
                    # If the card is public (known), skip
                    continue

            # print(f"Cards in player {player.name} hand after determinization: {player.hand}")

        # The remaining cards are randomly determinized as deck cards
        # print(f"Unknown cards before determinization: {unknown_cards}")
        random.shuffle(unknown_cards)

        # Assign the remaining unknown cards to the deck
        for card_to_assign in unknown_cards.copy():
            new_card = unknown_cards.pop(0)
            new_card.from_card(card_to_assign)
            new_card.is_public = True
            new_card.is_private = False
            new_card.is_unknown = False
            # Add to the beginning of the deck
            game_state.deck.cards.appendleft(new_card)

        # print(f"Unknown cards after determinization: {unknown_cards}")
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
            copied = self.determinize_with_model(copied)
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
            action_to_play, (W, N) = max(valid_total_ratings.items(), key=lambda x: x[1][0] / (x[1][1] + 1e-6))
            if game_state.print_info:
                print(f'{self} expects to not lose with {W/N*100:.2f}%')
        else:
            raise BaseException('Scoring type not implemented.')

        self.mcts = search_tree

        # assert action_to_play in game_state.allowed_plays(), f"Action {action_to_play} not allowed in {game_state.allowed_plays()}"

        return action_to_play