from moskaengine.players.abstract_player import AbstractPlayer
from moskaengine.mcts.mcts import MCTS
from moskaengine.utils.game_utils import state_as_vector
from moskaengine.research.model_training.train_model import CardPredictorMLP
from moskaengine.game.deck import Card, StandardDeck
import random
import torch

class DeterminizedMLPMCTS(AbstractPlayer):
    """


    """
    def __init__(self, name, model, device, deals=10, rollouts=100, expl_rate=0.7, scoring="win_rate"):
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


    def make_copy(self):
        new = DeterminizedMLPMCTS(self.name, self.model, self.device)
        new.hand = self.hand.copy()
        return new

    def determinize_with_model(self, game_state):
        """Determinize the game state using model predictions."""
        # Make each card in our hand public
        for card in game_state.player_to_play.hand:
            card.is_private = False
            card.is_public = True

        # Get unknown cards and their tuples (suit, value pairs)
        unknown_cards = list(game_state.get_non_public_cards())
        unknown_tuples = list(game_state.get_non_public_cards_tuples())

        # If there are no unknown cards, return the game state as is
        if not unknown_cards:
            print("Skipping determinization: No unknown cards.")
            return game_state

        # Encode the current game state into input vector
        input_state_vector, _ = state_as_vector(game_state)
        input_state_tensor = torch.tensor(input_state_vector, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Get model predictions
        num_opponents = len(game_state.players) - 1
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_state_tensor)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits).squeeze()

        # Reshape probabilities to match the number of opponents
        probs = probs.view(num_opponents, 52)

        # Get the number of hidden cards for each opponent
        opponents = [p for p in game_state.players if p != game_state.player_to_play]
        original_hand_sizes = {p: len(p.hand) for p in opponents}
        for player in opponents:
            # Ensure each player's hand is cleared of private cards
            player.hand = [card for card in player.hand if card.is_public]

        cards_needed_per_player = {p: original_hand_sizes[p] - len(p.hand) for p in opponents}

        print(f"Cards needed per player: {cards_needed_per_player}")

        reference_deck = StandardDeck(shuffle=False, perfect_info=True)

        # Compare the predicted cards with the actual cards
        # Get n most probable cards for each opponent
        for i in range(len(opponents)):
            top_values, top_indices = torch.topk(probs[i], int(cards_needed_per_player[opponents[i]]))
            print(f"Top {cards_needed_per_player[opponents[i]]} predicted cards for opponent {opponents[i]}: {top_indices.tolist()}")
            # # Create a mask for the probabilities
            # mask = torch.zeros_like(probs[i])
            # mask[top_indices] = probs[i][top_indices]
            # # Convert to binary mask (0 or 1)
            # mask = (mask > 0).float()
            # probs[i] = mask
        # Convert probabilities to a list of lists for easier handling
        probs = probs.cpu().numpy().tolist()
        print(f"Probabilities reshaped: {probs}")

        for i in range(len(opponents)):
            print(f"Predicted cards for opponent {opponents[i]}:")
            for j in range(52):
                if probs[i][j] == 1:
                    print(reference_deck.cards[j], end=", ")
            print()  # New line after each opponent's cards

        # Create a mapping from card indices (0-51) to actual cards in unknown_cards list
        reference_deck = game_state.card_collection
        card_mapping = {}

        for i, ref_card in enumerate(reference_deck):
            for unknown_card in unknown_cards:
                if (ref_card.suit == unknown_card.suit and
                        ref_card.value == unknown_card.value):
                    card_mapping[i] = unknown_card
                    break

        # Assign cards to players based on probabilities
        assigned_cards = set()

        # For each player, assign cards with highest probabilities
        for idx, player in enumerate(opponents):
            if idx >= len(probs):  # Safety check
                continue

            num_cards_needed = cards_needed_per_player[player]
            if num_cards_needed <= 0:
                continue

            player_probs = probs[idx]

            print(f"Player probs: {player_probs}")

            # Create pairs of (card_idx, probability)
            card_prob_pairs = [(i, prob) for i, prob in enumerate(player_probs)
                               if i in card_mapping and card_mapping[i] not in assigned_cards]

            print(f"Card-probability pairs for {player.name}: {card_prob_pairs}")
            # Sort by probability (highest first)
            card_prob_pairs.sort(key=lambda x: x[1], reverse=True)

            # Assign cards to this player
            for _ in range(min(num_cards_needed, len(card_prob_pairs))):
                if not card_prob_pairs:
                    break

                card_idx, _ = card_prob_pairs.pop(0)
                card_to_assign = card_mapping[card_idx]

                # Skip if already assigned
                if card_to_assign in assigned_cards:
                    continue

                # Create a new card to avoid reference issues
                new_card = Card()
                new_card.from_card(card_to_assign)
                new_card.is_public = True
                new_card.is_private = False
                new_card.is_unknown = False

                # Add to player's hand
                player.hand.append(new_card)
                print(f"Assigned {new_card} to {player.name}")

                # Mark as assigned
                assigned_cards.add(card_to_assign)

        # Assign any remaining cards that weren't assigned based on probabilities
        unassigned_cards = [card for card in unknown_cards if card not in assigned_cards]

        # Shuffle for randomness
        random.shuffle(unassigned_cards)

        # Distribute remaining cards to players who still need them
        for player in opponents:
            cards_still_needed = original_hand_sizes[player] - len(player.hand)

            for _ in range(min(cards_still_needed, len(unassigned_cards))):
                if not unassigned_cards:
                    break

                card = unassigned_cards.pop(0)

                # Create a new card to avoid reference issues
                new_card = Card()
                new_card.from_card(card)
                new_card.is_public = True
                new_card.is_private = False
                new_card.is_unknown = False

                # Add to player's hand
                player.hand.append(new_card)

                # Mark as assigned
                assigned_cards.add(card)

        # Any remaining cards stay in the deck as unknown cards
        # But we need to assign them concrete values for the simulation
        remaining_cards = [card for card in unknown_cards if card not in assigned_cards]
        remaining_tuples = list(unknown_tuples)
        random.shuffle(remaining_tuples)

        for card in remaining_cards:
            if remaining_tuples:
                suit, value = remaining_tuples.pop(0)
                card.from_suit_value(suit, value)
                card.is_public = True
                card.is_private = False
                card.is_unknown = False

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