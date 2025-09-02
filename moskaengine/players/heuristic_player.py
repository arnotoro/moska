import random

# Moskaengine imports
from .abstract_player import AbstractPlayer

class HeuristicPlayer(AbstractPlayer):
    def make_copy(self):
        new = HeuristicPlayer(self.name)
        new.hand = self.hand.copy()
        return new

    def choose_action(self, game_state):
        # Make hand cards known to the player
        self.make_cards_known(game_state)

        # Get the allowed actions
        allowed_actions = game_state.allowed_plays()

        # If only one action is allowed, return it
        if len(allowed_actions) == 1:
            return allowed_actions[0]

        # Occasionally, play a random action to add some variability
        if random.random() < 0.05:
            return random.choice(allowed_actions)

        scored_moves = []
        for move in allowed_actions:
            # Add a small random factor to each score for tiebreaking
            randomness = random.uniform(-0.5, 0.5)
            score = self._evaluate_move(move, game_state) + randomness
            scored_moves.append((move, score))

        # Sort the scored moves by score
        scored_moves.sort(key=lambda x: x[1], reverse=True)

        return scored_moves[0][0]

    def _evaluate_move(self, possible_move, game_state):
        """
        Evaluate the possible move and return a score.
        The higher the score, generally, the stronger the move.
        """
        score = 0
        move = possible_move[0]

        # print(possible_move)
        if possible_move[1] is not None:
            # If the turn has a playable card, assign it
            playable_card = possible_move[1][0] if isinstance(possible_move[1], tuple) else possible_move[1]
            killable_card = possible_move[1][1] if isinstance(possible_move[1], tuple) and move != "ThrowCards" else None

            # If the cards on the table are lower than 7 in value, we should try to kill them
            if killable_card is not None and killable_card.value < 7:
                # Generally play the lowest card possible
                score += self._score_from_card(playable_card, game_state) * 2
            elif killable_card is not None and killable_card.value >= 7:
                # If the card's value is higher than 7, we should be more conservative
                score += self._score_from_card(playable_card, game_state) * 1
            else:
                if move == "PlayFromDeck":
                    # We don't know the card to play from deck so if we can play a card from our hand, we should
                    # Give this move a negative score
                    score -= 10
                else:
                    # Attacking
                    score += self._score_from_card(playable_card, game_state)

        else:
            if move == "ThrowCards":
                score -= 10
            else:
                # This is only true if the turn is to pick up cards from the table, often a bad move
                score -= 100

        # print(possible_move, playable_card, score)

        return score

    def _score_from_card(self, card, game_state):
        """
        Score the card based on its value and suit.
        The higher the score, the better the card.
        """
        card_score = 0
        trump_suit = game_state.trump_card.suit
        deck_left = len(game_state.deck)

        # Add a small random factor to each card's score (±10%)
        randomness = random.uniform(0.9, 1.1)

        if card.suit == trump_suit:
            card_score = (5 + card.value) * 1 * randomness
        else:
            card_score += (15 - card.value) * 3 * randomness

        # Consider deck size - be more conservative when deck is small
        if deck_left < 10:
            # Save trump cards for later
            if card.suit == trump_suit:
                card_score *= 0.8  # Reduce score to discourage playing trumps early

        return card_score