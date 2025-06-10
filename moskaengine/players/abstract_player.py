from abc import ABC, abstractmethod
import random

from moskaengine.utils.card_utils import choose_random

class AbstractPlayer(ABC):

    def __init__(self, name):
        self.name = name
        self.hand = []
        self.id = None
    
    def __str__(self):
        return f"{self.name}"

    def fill_hand(self, deck):
        """Fill the hand with cards from the deck"""
        draw = 6 - len(self.hand)

        if draw <= 0:
            return deck

        # Draw the cards from the deck
        for _ in range(draw):
            # Check if the deck is empty, if so, don't draw
            if len(deck) == 0:
                return deck
            
            # Otherwise, draw the cards
            card_drawn = deck.pop(1)[0]
            self.hand.append(card_drawn)

        return deck
    
    def make_cards_known(self, game_state):
        """Make the cards in the hand known to the player"""
        for card in self.hand:

            # All cards must be unknown
            if card.is_unknown:
                unknown = game_state.get_unknown_cards()
                if game_state.computer_shuffle:
                    card.from_suit_value(*choose_random(unknown))
                else:
                    print(f"{self} has drawn a card")
                    card.from_input(unknown)
                card.is_private = True
    
    def possible_card_plays(self, non_public_cards):
        """Returns the (suit, value) pairs this person can play from his hand"""
        possible = set()

        for card in self.hand:
            if card.is_unknown:
                possible.update(non_public_cards)
            else:
                possible.add(card)

        return possible
    
    def discard_card(self, game_state, suit, value, remove = True):
        """Discard a card from the hand with the given suit and value"""
        # TODO: Comment this
        for idx, card in enumerate(self.hand):
            if card.is_unknown:
                if (suit, value) in game_state.get_non_public_cards_tuples():
                    if (suit, value) not in [(c.suit, c.value) for c in self.hand if not c.is_unknown]:
                        card.suit = suit
                        card.value = value
                        break
            else:
                if card.suit == suit and card.value == value:
                    break
        else:
            raise BaseException('Card not possible to discard')
        
        if remove:
            card_played = self.hand.pop(idx)
        else:
            card_played = self.hand[idx]

        card_played.is_unknown = False
        card_played.is_private = False
        card_played.is_public = True

        return card_played

    def can_throw(self, fallback_identities, cards):
        """Check if the player can throw the given cards. Fallback identities are the options for unknown cards."""
        cards_set = set(cards)  # Set of target cards for fast lookups
        fallback = 0
        poss = []

        # Classify the cards in hand into known and unknown cards
        known_cards = set()
        for card in self.hand:
            if card.is_unknown:
                fallback += 1
            else:
                known_cards.add(card)  # Store the card directly

        # Match known cards in hand with the cards to throw
        for card in cards:
            if card in known_cards:
                poss.append({card})  # Known valid card

        # If we don't have enough known cards to satisfy the throw, add fallback identities
        remaining_needed = len(cards) - len(poss)
        if remaining_needed > 0:
            fallback_identities_needed = min(fallback, remaining_needed)
            poss += [set(fallback_identities) for _ in range(fallback_identities_needed)]

        # Check if we can match all cards in poss, either by direct match or fallback
        remaining_cards = set(cards)  # We need to match each card in cards
        poss_set = set(card for subset in poss for card in subset)  # Flatten poss into a set for faster checking

        # If poss contains all the needed cards, we can throw
        if remaining_cards.issubset(poss_set):
            return True
        return False

    def determinize_hand(self, game_state):
        """Returns a possible determinization for the hand of this player"""
        # First, we make every card in our hand public
        unknown_cards = []
        for card in self.hand:
            if card.is_unknown:
                unknown_cards.append(card)
            else:
                card.is_private = False
                card.is_public = True

        # Check if we need to do anything
        if len(unknown_cards) > 0:
            # We pop from all the unknown cards as possible cards
            unknown = list(game_state.get_non_public_cards_tuples())
            for unknown_card in unknown_cards:
                suit, value = unknown.pop(random.randint(0, len(unknown)-1))
                unknown_card.from_suit_value(suit, value)
                unknown_card.is_public = True


    @abstractmethod
    def choose_action(self, game_state):
        """Choose an action for the player"""
        pass

    @abstractmethod
    def make_copy(self):
        """Make a copy of the player without copying the hand"""
        pass             
