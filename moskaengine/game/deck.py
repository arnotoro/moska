from collections import deque
from itertools import product
from random import randint, seed, shuffle

# Possible values and suits for a card
CARD_VALUES = tuple(range(2, 15))
CARD_VALUES_SYMBOLS = {11: "J", 12: "Q", 13: "K", 14: "A"}

CARD_SUITS = tuple(range(1, 5))
CARD_SUITS_SYMBOLS = {1: "♣", 2: "♠", 3: "♥", 4: "♦", "X": "X"}


def suit_to_symbol(suit):
    # Convert suit to symbol based on int value
    return CARD_SUITS_SYMBOLS[suit] if suit in CARD_SUITS_SYMBOLS else str(suit)

def value_to_letter(value):
    return CARD_VALUES_SYMBOLS[value] if value in CARD_VALUES_SYMBOLS else str(value)


class Card:
    """ A regular Moska playing card.

    Can be unknown to all, private to holder or public to all.
    """
    suit = None
    value = None
    trump_suit = None
    kopled = False

    # Always one of the following is true
    is_public = False # Does everyone know the card
    is_private = False # Does the holder know the card
    is_unknown = True # Does no one know the card

    def __init__(self, value, suit, kopled = False, trump_suit = None):
        self.value = value
        self.suit = suit
        self.kopled = kopled
        self.trump_suit = trump_suit
        self.is_public = False
        self.is_private = False
        self.is_unknown = True

    def __hash__(self):
        return hash((self.value, self.suit))

    def __repr__(self):
        if self.is_unknown:
            return "-X"
        else:
            return str(f"{value_to_letter(self.value)}{suit_to_symbol(self.suit)}")

    def __len__(self):
        # TODO: This is a bit of a hack, but it works for now
        return 1

    def __eq__(self, other):
        return self.suit == other.suit and self.value == other.value

    def __lt__(self, other):
        return self.value < other.value

    def __str__(self):
        # if self.is_unknown:
        #     return "-X"
        # else:
        #     if self.is_private:
        #         string = "P"
        #     else:
        #         string = "A"
        #
        #     assert self.suit is not None
        #     string += suit_to_symbol(self.suit)
        #     string += value_to_letter(self.value)
        #     return string

        return str(f"{value_to_letter(self.value)}{suit_to_symbol(self.suit)}")

    def reset(self):
        """Reset values or make unknown (useful for keeping same memory address)"""
        # Keeping the same memory address is adamant for this program
        # (not only the copy) as otherwise self.card_collection would
        # not be a collection of the cards anymore.
        self.suit = None
        self.value = None
        self.trump_suit = None
        self.is_public = False
        self.is_private = False
        self.is_unknown = True

    def make_copy(self):
        """Returns a copy of the card"""
        new = Card()
        new.suit = self.suit
        new.value = self.value
        new.trump_suit = self.trump_suit
        new.is_public = self.is_public
        new.is_private = self.is_private
        new.is_unknown = self.is_unknown
        return new

    def is_trump(self):
        """Checks if the current card is a trump card or not"""
        return self.suit == self.trump_suit

    def from_input(self, possible):
        """Get the suit and value of the card from the input

        Args:
            - possible: The (suit, value) allowed options.

        Remark: remember to set private/public permissions
        """

        # We cannot overwrite a current value
        assert self.is_unknown

        while True:
            suit = eval(input('Suit of the card [♣♠♥♦]: '))
            value = eval(input('Value of the card [23456789*JQKA]: '))
            if (suit, value) in possible:
                break
            print(f'Not valid, try again (one of {" ".join("♣♠♥♦"[i[0]] + "23456789*JQKA"[i[1]] for i in possible)})')
        self.suit = suit
        self.value = value
        self.is_unknown = False

    def from_suit_value(self, suit, value):
        """Set the suit and value of the card

        Remark: remember to set private/public permissions
        """
        # We cannot overwrite a current value
        assert self.is_unknown
        self.suit = suit
        self.value = value
        self.is_unknown = False

class StandardDeck:
    "A deck of playing cards."

    seed_value : int = None
    cards : deque = None

    def __init__(self, shuffle = True, seed_value = None, perfect_info = False):
        self.seed_value = seed_value if seed_value else randint(0, 100_000_000)
        seed(self.seed_value)
        self.cards = deque(
            Card(value, suit) for value, suit in product(
                CARD_VALUES, CARD_SUITS
            )
        )
        if shuffle:
            self.shuffle()
        if perfect_info:
            for card in self.cards:
                card.is_unknown = False
                card.is_private = False
                card.is_public = True

    def __len__(self):
        return len(self.cards)

    def __str__ (self):
        return f"Deck has {len(self)} cards left"

    def __repr__(self):
        s = ""
        for card in self.cards:
            s += str(card) + " "
        return s

    def __hash__(self):
        return hash(tuple(self.cards))
    
    def shuffle(self):
        shuffle(self.cards)

    def pop(self, n):
        "Pop (draw) n cards from the top of the deck."
        hand = []
        if not self.cards or n <= 0:
            return []

        for _ in range(min(len(self), n)):
            card = self.cards.popleft()
            # Make cards known to the player
            card.is_unknown = False
            card.is_private = True
            card.is_public = False
            hand.append(card)

        # hand = [self.cards.popleft() for _ in range(min(len(self), n))]
        return hand
    
    def place_bottom(self, card):
        "Place a card at the bottom of the deck."
        self.cards.append(card)

    def place_top(self, card):
        "Place a card at the top of the deck."
        self.cards.appendleft(card)

    def copy(self):
        "Returns a copy of the deck."
        new_deck = StandardDeck(shuffle=False, seed_value=self.seed_value)
        new_deck.cards = self.cards.copy()
        return new_deck