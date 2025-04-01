from abc import ABC, abstractmethod
import random

from moskaengine.utils.card_utils import choose_random

class AbstractPlayer(ABC):

    def __init__(self, name):
        self.name = name
        
        # TODO: cards should be a list of Card objects
        self.hand = []

    
    def __str__(self):
        return f"Player {self.name}"

    def fill_hand(self, deck):
        """Fill the hand with cards from the deck"""
        draw = 6 - len(self.hand)

        if draw <= 0:
            return deck
        
        for _ in range(draw):
            
            # Check if the deck is empty
            if len(deck) == 0:
                return deck
            
            # Otherwise, draw the cards
            card_drawn = deck.pop(0)

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
                    print(f"self has drawn a card")
                    card.from_input(unknown)

                card.is_private = True
    
    def possible_card_plays(self, non_public_cards):
        """Get the possible card plays for the player"""
        possible = set()

        for card in self.hand:
            if card.is_unknown:
                possible.update(non_public_cards)
            else:
                possible.add((card.suit, card.value))

        return possible
    
    def discard_card(self, game_state, suit, value, remove = True):
        """Discard a card from the hand with the given suit and value"""

        for id, card in enumerate(self.hand):
            if card.is_unknown:
                
                if (suit, value) in game_state.get_non_public_cards():
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
            card_played = self.hand.pop(id)

        else:
            card_played = self.hand[id]

        card_played.is_unknown = False
        card_played.is_private = False
        card_played.is_public = True

        return card_played
    
    def can_throw(self, fallback_identities, cards):
        """Check if the player can throw the given cards
        Fallback identities are the options of the cards if it is unknown
        """

        poss = []
        cards_set = set(cards)
        fallback = 0
        for card in self.hand:
            if card.is_unknown:
                fallback += 1
            else:
                identity = (card.suit, card.value)

                # Check if this card has an identity that match a card in cards
                if identity in cards_set:
                    poss.append({identity})
        # Easy case, poss is not big enough to consist of len(cards) cards
        # if len(poss) < len(cards):
        if len(poss) + fallback < len(cards):
            return False
        # Add fallbacks, the minimum amount needed
        poss += [fallback_identities.copy() for _ in range(min(fallback, len(cards)))]
        ### We need to check if we can play cards, having poss
        # # Easy case, one of the cards is not in poss
        # p = set()
        # for i in poss:
        #     p = p.union(i)
        # for c in cards:
        #     if c not in p:
        #         return False
        # Greedy approach: take the nth card from the first allowed poss
        poss2 = [i.copy() for i in poss]
        for c in cards:
            for idx, p in enumerate(poss2):
                if c in p:
                    poss2.pop(idx)
                    break
            else:
                return False
        return True

        # # Otherwise iterating through all options
        # # -> takes a long time
        # for _ in range(len(poss) - len(cards)):
        #     cards.append(0)
        # for perm in permutations(cards, r=len(cards)):
        #     # print(perm, poss)
        #     for idx, card in enumerate(perm):
        #         if card != 0 and card not in poss[idx]:
        #             break
        #     else:
        #         return True
        # return False

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
            unknown = list(game_state.get_non_public_cards())
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
