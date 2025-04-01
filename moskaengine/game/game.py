from moskaengine.game.deck import StandardDeck, CARD_SUITS, CARD_VALUES
from moskaengine.utils.card_utils import choose_random




class MoskaGame:
    """
    The class for representing a game of Moska.

    Args:
        - players:          The players in the game as classes
        - computer_shuffle: If set to True, the computer shuffles and the
                                game is virtual, if set to False, you have
                                to shuffle and draw the cards irl and tell
                                it to the computer.
        - main_attacker:    The name of the starting attacker
    """

    is_end_state = False
    loser = None  # The loser of the game once known
    print_info = True

    def __init__(self, players, computer_shuffle, main_attacker, seed_value = None, do_init = True, print_info = True):
        if not do_init:
            return
        
        self.players = players
        self.computer_shuffle = computer_shuffle
        self.main_attacker = main_attacker
        self.seed_value = seed_value

        # Initialize the deck
        self.deck = StandardDeck(seed_value = self.seed_value)

        self.all_cards = {(suit, value) for suit in CARD_SUITS for value in CARD_VALUES}

        self.card_collection = self.deck.copy()

        self.history = []


        # Initialize the bottom card of the deck
        if computer_shuffle:
            unknown = self.get_unknown_cards()

            assert self.all_cards == unknown

            self.deck[-1].from_suit_value(*choose_random(unknown))
            self.deck[-1].is_public = True

        else:
            # We must shuffle the cards irl
            print("Specify the suit and value of the trump card:")
            self.deck[-1].from_input(self.all_cards)
            self.deck[-1].is_public = True

        # Display bottom card
        if self.print_info:
             print(f'The bottom card is {str(self.deck[-1])[1:]}')
        

        # Set trump
        for card in self.deck:
            card.trump_suit = self.deck[-1].suit

        # Initialize the players
        for player in self.players:
            # The suit and value of the card is not necessary at this point
            # only if the player is not Human the suit and value of the card
            # must be added whenever all possible actions are listed.
            self.deck = player.fill_hand(self.deck)

        # Initialize the attacking player
        self.new_attack(main_attacker)
        
    
    def get_unknown_cards(self):
        """Returns the unknown cards in the deck"""
        remove = {(c.suit, c.value) for c in self.deck if not c.is_unknown}
        return self.all_cards - remove
    
    def get_non_public_cards(self):
        """Returns the (suit, value) pairs of all unknown cards + private cards"""
        # remove = set()
        # for card in self.card_collection:
        #     if card.is_public:
        #         remove.add((card.suit, card.value))
        remove = {(c.suit, c.value) for c in self.card_collection if c.is_public}
        return self.all_cards - remove