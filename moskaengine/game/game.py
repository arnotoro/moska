from moskaengine.game.deck import Card, suit_to_symbol, StandardDeck
from moskaengine.utils.card_utils import choose_random
from moskaengine.players.human_player import Human
from moskaengine.utils.card_utils import basic_repr_game


from itertools import combinations
from collections import defaultdict




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
    trump_card = None  # The trump card

    def __init__(self, players, computer_shuffle, main_attacker, do_init = True, print_info = True, perfect_info = False):

        if not do_init:
            return

        self.players = players
        self.computer_shuffle = computer_shuffle
        self.print_info = print_info
        self.perfect_info = perfect_info

        # Initialize the player variants
        self.attackers = None
        self.defender = None
        self.current_attacker = None
        self.draw_undefended = False

        # Initialize the deck
        self.deck = StandardDeck(shuffle=True)
        # print(repr(self.deck))

        # Initialize
        self.all_cards = {(suit, value) for suit in range(1, 5) for value in range(2, 15)}

        self.card_collection = self.deck.copy()

        # Keep track of actions played in the game
        self.history = []

        # Players draw their hand
        for player in self.players:
            player.hand = self.deck.pop(6)
            # print(f"{player} has drawn a hand: {player.hand}")


        # print(f"Deck after drawing hands: {repr(self.deck)}")



        if computer_shuffle:
            unknown = self.get_unknown_cards()

            # Draw trump card
            self.trump_card = self.deck.pop(1)[0]
            self.trump_card.is_public = True
            print(f"Trump card: {self.trump_card}")

            # Check if a player has the equivalent of 2 of trump and if so, switch the card
            for player in self.players:
                # Find the 2 of trump card in player's hand (if it exists)
                for card in player.hand:
                    if card.suit == self.trump_card.suit and card.value == 2:
                        player.hand.remove(card)
                        player.hand.append(self.trump_card)
                        self.deck.place_bottom(card)
                        print(f"{player} has switched the trump card ({self.trump_card}) with their 2 of trump card ({card})")

                        card.is_public = True
                        card.is_unknown = False
                        self.trump_card.is_public = True
                        self.trump_card.is_unknown = False

                        self.trump_card = card
                        break
        else:
            # TODO: Not implemented correctly yet
            # We must shuffle the cards irl
            print("Specify the suit and value of the trump card:")
            self.deck[-1].from_input(self.all_cards)
            self.deck[-1].is_public = True

        # TODO: Not sure if this is needed, if is, change the logic to check for trump card. This seems odd.
        for card in self.deck.cards:
            card.trump_suit = self.trump_card.suit
            if card.is_public:
                print(card)

        # Initialize the players
        for player in self.players:
            # The suit and value of the card is not necessary at this point
            # only if the player is not Human the suit and value of the card
            # must be added whenever all possible actions are listed.
            self.deck = player.fill_hand(self.deck)


        # Initialize the attacking player
        self.new_attack(main_attacker)


    def get_unknown_cards(self):
        """Returns the (suit, value) pairs of all unknown cards"""
        # remove = set()
        # for card in self.card_collection:
        #     if not card.is_unknown:
        #         remove.add((card.suit, card.value))
        remove = {(c.suit, c.value) for c in self.card_collection.cards if not c.is_unknown}
        return self.all_cards - remove

    def get_non_public_cards(self):
        """Returns the (suit, value) pairs of all unknown cards + private cards"""
        # remove = set()
        # for card in self.card_collection:
        #     if card.is_public:
        #         remove.add((card.suit, card.value))
        remove = {(c.suit, c.value) for c in self.card_collection.cards if c.is_public}
        return self.all_cards - remove

    def new_attack(self, main_attacker):
        """Initialize new attack move with the main_attacker as starting player."""

        # Search for the main attacker
        active_players = len(self.players)

        for id, player in enumerate(self.players):
            if player.name == main_attacker:
                break
        else:
            raise "Player not found"

        # We initialize the players starting from the main attacker
        # A person is still in the game whenever his hand is not empty or if he can draw a card.
        self.attackers = [person for i in range(id, id+active_players)
                            if (len((person := self.players[i%active_players]).hand) > 0 or
                                len(self.deck) > 0)]

        # Check if the game has ended
        if len(self.attackers) == 0:
            # There are no attackers left, the last card got defended, so the last defender lost
            self.is_end_state = True
            self.loser = self.defender
        elif len(self.attackers) == 1:
            # There is only one player left in the game, the loser
            self.is_end_state = True
            self.loser = self.attackers[0]
        else:
            # The game continues
            self.defender = self.attackers.pop(1)

            # The main attacker may start off as the initiating player
            self.current_attacker = 0
            self.player_to_play = self.attackers[0]

            # People must draw cards according to the draw order
            self.draw_order = self.attackers + [self.defender]

            # The person to the left of the defender if he takes
            self.attacker_to_start_throwing = None  # To check if all have thrown

            # To check if all attackers passed
            self.last_played_attacker = None

            # All the trumps that were used to reflect this turn (they lost their life)
            self.reflected_trumps = []

        # The action to perform
        self.current_action = 'Attack'
        # Succesfully defended cards as (attack, defend) pairs
        self.cards_killed = []
        self.cards_to_defend = []

    def allowed_plays(self):
        """Allowed plays for the current attacker"""
        player = self.player_to_play
        action = self.current_action
        possible_actions = []

        if action == "Attack":
            attacker = self.attackers[self.current_attacker]
            assert attacker == player

            # List all possible attacks
            possible_plays = player.possible_card_plays(self.get_non_public_cards())

            # NOTE: here game is already initiated
            if len(self.cards_killed) > 0:
                # There is a pair on the table, the player can pass on attacking
                possible_actions.append(('PassAttack', None))

                # If no pass, we play cards with same values as those that are played to table
                values_on_table = {card.value for pair in self.cards_killed for card in pair}
                possible_plays = [card for card in possible_plays if card[1] in values_on_table]

            # Group cards by value
            cards_by_value = {}
            for suit, value in possible_plays:
                if value not in cards_by_value:
                    cards_by_value[value] = []
                cards_by_value[value].append((suit, value))

            defender_hand_size = len(self.defender.hand)
            current_attack_count = len(self.cards_to_defend)
            max_additional_attacks = defender_hand_size - current_attack_count

            # Check if table has space to play new cards for the defender
            if max_additional_attacks > 0:
                # Single card attacks
                for suit, value in possible_plays:
                    possible_actions.append(('Attack', (suit, value)))

                # A combination card attack
                for value, cards in cards_by_value.items():
                    if len(cards) > 1:
                        import itertools

                        for i in range(2, min(len(cards) + 1, max_additional_attacks + 1)):
                            for combination in itertools.combinations(cards, i):
                                print(f"The allowed combinations: {combination}")
                                if len(combination) <= max_additional_attacks:
                                    possible_actions.append(('Attack', combination))

        elif action == "Defend":
            # Iterate through the cards in the hand to see which ones can be played
            play_options = defaultdict(int)

            # Get all cards that are to be defended
            to_defend = self.cards_to_defend[0]
            # TODO: make it possible to play multiple cards to defend at the same turn




            # TODO: Implement the logic for the defender to Koplata from the deck and to play to self.

            for card in player.hand:
                if card.is_unknown:
                    identities = self.get_non_public_cards() - {(card.suit, card.value) for card in player.hand if not card.is_unknown}
                else:
                    identities = {(card.suit, card.value)}
                # Note: Identities is a list of cards that are in the hand

                defend = []

                ## Defending
                for suit, value in identities:
                    # Check if the card can defend the to_defend card
                    if suit == self.trump_card.suit and not to_defend.is_trump():
                        # Can always defend with trump on a non-trump card
                        defend.append(('Defend', (suit, value)))

                    if suit == to_defend.suit:
                        # Can always defend the same suit with higher value
                        if value > to_defend.value:
                            defend.append(('Defend', (suit, value)))

                # Add all options together to possible actions.

                for action in defend:
                    play_options[action] += 1 / len(defend)


            ## Koplaus
            # Allowed if there is not kopled card on the table
            if not any(defend_card.kopled for defend_card in self.cards_to_defend):
                # Check if there are cards in the deck
                if len(self.deck) > 0:
                    next_card = self.deck.cards[0]
                    print(f"Next card: {next_card} {to_defend.trump_suit}")

                    # Check if the kopled card could fall a card on the table
                    if next_card.suit == to_defend.suit and next_card.value > to_defend.value:
                        possible_actions.append(('PlayFromDeck', (next_card.suit, next_card.value), 1/2, False))
                    elif next_card.suit == self.trump_card.suit and not to_defend.is_trump():
                        possible_actions.append(('PlayFromDeck', (next_card.suit, next_card.value), 1/2, False))
                    else:
                        # If the kopled card can't fall a card on the table, it must be added to the cards_to_defend
                        possible_actions.append(('PlayFromDeck', (next_card.suit, next_card.value), 1/2, True))

            # Restructure playing options

            for key, weight in play_options.items():
                possible_actions.append(key + (weight,))

            # You can also pick up only the cards on the table if there is defended cards
            if self.cards_killed:
                possible_actions.append(('TakeDefend', None, 1/2))

            # As the defender you can also always pick up all the cards
            possible_actions.append(('TakeAll', None, 1/2))


        elif action == "ThrowCards":
            # Essentially PlayToOther

            possible_throws = player.possible_card_plays(self.get_non_public_cards())

            # You can only throw cards with the same value as those on the table
            values_on_table = {card.value for pair in self.cards_killed for card in pair}

            values_on_table.update({card.value for card in self.cards_to_defend})

            possible_throws = {card for card in possible_throws if card[1] in values_on_table}

            available_throws = len(self.defender.hand) - len(self.cards_to_defend)

            # If 0 cards thrown
            possible_actions.append(('ThrowCards', (None, )))

            # If more than 0 cards thrown
            max_throws = min(available_throws, len(possible_throws), len(player.hand))

            if max_throws > 0:
                fallback_identities = self.get_non_public_cards()

            for throw in range(1, max_throws + 1):
                # Any combination works
                for option in combinations(possible_throws, r = throw):
                    if player.can_throw(fallback_identities, list(option)):
                        possible_actions.append(('ThrowCards', option))

        else:
            raise ValueError(f"Unknown action {action}")

        if len(possible_actions) == 0:
            raise BaseException(f"No possible actions for {player.name} in {action}")

        print(f"Found possible actions: {possible_actions}")
        return possible_actions

    def get_id(self):
        return hash(tuple(self.history))

    def execute_action(self, action):
        # Add to history
        self.history.append(action)

        if action[0] == 'Attack':
            # The attacking move i.e. the first play to an empty table
            print(action[1], len(action[1]))
            if isinstance(action[1], list):
                # Multiple cards case
                for suit, value in action[1]:
                    card_played = self.player_to_play.discard_card(self, suit, value)
                    self.cards_to_defend.append(card_played)
            else:
                suit, value = action[1]
                card_played = self.player_to_play.discard_card(self, suit, value)
                self.cards_to_defend.append(card_played)

            # Update values
            self.last_played_attacker = self.player_to_play
            self.player_to_play = self.defender
            self.current_action = 'Defend'

        elif action[0] == 'Defend':
            # TODO: This needs to be changed for the card to be choosable
            # The defending move from the target player for the current turn
            card_defended = self.cards_to_defend.pop(0)

            suit, value = action[1]
            card_played = self.player_to_play.discard_card(self, suit, value)
            self.cards_killed += [(card_defended, card_played)]

            if len(self.cards_to_defend) == 0:
                # No more cards to defend, switch to attacking
                self.player_to_play = self.attackers[self.current_attacker
                ]
                self.current_action = 'Attack'

        elif action[0] == 'PlayFromDeck':
            # Draw a card from the deck
            print(f"PlayFromDeck: {action}")
            kopled_card = self.deck.pop(1)[0]
            print(kopled_card)
            kopled_card.kopled = True
            kopled_card.is_public = True
            kopled_card.is_unknown = False

            # If this is true, the card can't fall a card on the table, and therefore it is added to the cards_to_defend
            if action[1]:
                self.cards_to_defend.append(kopled_card)
            # Otherwise, the card can fall a card on the table
            # TODO: This should be choosable by the player, for now it is the first card
            else:
                card_defended = self.cards_to_defend.pop(0)
                self.cards_killed += [(card_defended, kopled_card)]

            # If all the cards are defended, the game continues accordingly
            if len(self.cards_to_defend) == 0:
                # No more cards to defend, switch to attacking
                self.player_to_play = self.attackers[self.current_attacker
                ]
                self.current_action = 'Attack'


        elif action[0] == 'TakeAll':
            # Take cards from the table, should be only available for the defender
            assert self.player_to_play == self.defender
            self.current_action = 'ThrowCards'
            self.player_to_play = self.attackers[self.current_attacker]
            self.attacker_to_start_throwing = self.current_attacker

        elif action[0] == 'TakeDefend':
            # Take cards only the non defended cards from the table
            assert self.player_to_play == self.defender
            self.draw_undefended = True
            self.current_action = 'ThrowCards'
            self.player_to_play = self.attackers[self.current_attacker]
            self.attacker_to_start_throwing = self.current_attacker


        elif action[0] == 'ThrowCards':
            # PlayToOther i.e. play to table for the defender to fall
            if action[1][0] is not None:

                cards_to_throw = action[1]

                for suit, value in cards_to_throw:
                    card_played = self.player_to_play.discard_card(self, suit, value)
                    self.cards_to_defend.append(card_played)

            # Increment attacker and player to play
            self.current_attacker = (self.current_attacker + 1) % len(self.attackers)
            self.player_to_play = self.attackers[self.current_attacker]


            # Check if everybody got a chance to throw
            if self.player_to_play == self.attackers[self.attacker_to_start_throwing]:
                if self.draw_undefended:
                    # The defender takes the non defended cards and the new main attacker is the one to the left of the defender
                    cards_not_defended = self.cards_to_defend
                    self.defender.hand += cards_not_defended
                    self.draw_undefended = False
                else:
                    cards_on_table = [card for pair in self.cards_killed for card in pair]
                    cards_on_table += self.cards_to_defend
                    self.defender.hand += cards_on_table

                for player in self.draw_order:
                    self.deck = player.fill_hand(self.deck)

                # Defender takes the cards and the new main attacker is the one to the left of the defender
                self.new_attack(self.attackers[1 % len(self.attackers)].name)

        elif action[0] == 'PassAttack':
            # Pass on attacking i.e. skip
            self.current_attacker = (self.current_attacker + 1) % len(self.attackers)
            self.player_to_play = self.attackers[self.current_attacker]

            # Check if no one wants to attack
            if self.player_to_play == self.last_played_attacker:
                # The defender defended successfully
                for player in self.draw_order:

                    self.deck = player.fill_hand(self.deck)
                    print(self.deck, player.hand)

                assert self.cards_to_defend == []

                self.new_attack(self.defender.name)

        elif action[0] == 'Reflect':
            card_played = self.defender.discard_card(self, action[1][0], action[1][1])

            # The new defender sits left of the current defender, the main attacker
            # stays the same and the current cards need to be defended
            self.last_played_attacker = self.player_to_play
            num_old_attackers = len(self.attackers)
            new_defender = self.attackers.pop(1 % num_old_attackers)
            self.attackers.insert(1 % num_old_attackers, self.defender)
            self.defender = new_defender
            self.draw_order = self.attackers + [self.defender]

            # The current attacker
            self.attackers = self.attackers[1:] + [self.attackers[0]]
            self.cards_to_defend.append(card_played)
            self.current_action = 'Defend'
            self.player_to_play = self.defender

        elif action[0] == 'ReflectTrump':
            # TODO: Check if this is needed for Moska
            # By only having to show the trump you can reflect the cards
            suit, value = action[1]

            # You must be the defender to do this
            assert self.player_to_play == self.defender

            # Everyone now knows you have that trump card, but you do not lose the card
            self.player_to_play.discard_card(self, suit, value, remove = False)

            # This card loses its ability to reflect for the rest of this trick
            self.reflected_trumps.append((suit, value))

            # The new defender sits left of the current defender,
            # the main attacker stays the same
            # and the current cards need to be defended
            self.last_played_attacker = self.player_to_play
            num_old_attackers = len(self.attackers)
            new_defender = self.attackers.pop(1 % num_old_attackers)
            self.attackers.insert(1 % num_old_attackers, self.defender)
            self.defender = new_defender
            self.draw_order = self.attackers + [self.defender]

            # The current attacker
            self.attackers = self.attackers[1:] + [self.attackers[0]]
            self.current_action = 'Defend'
            self.player_to_play = self.defender

        else:
            raise NotImplementedError('Action to execute not implemented.')


    def make_deepcopy(self):
        """Returns a deepcopy of the MoskaGame, faster than deepcopy"""
        ### Deepcopy code for checks
        # from copy import deepcopy
        # new = deepcopy(self)
        # new.print_info = False
        # return new
        ### Faster code
        new = MoskaGame(0, 0, 0, False)
        players = [p.make_copy() for p in self.players]

        ### We copy all the players in all the places
        new.players = []
        player_ids = {}  # dict of all the players with their copy
        for p in self.players:
            copy_p = p.make_copy()
            player_ids[id(p)] = copy_p
            new.players.append(copy_p)

        new.attackers = [player_ids[id(p)] for p in self.attackers]
        new.draw_order = [player_ids[id(p)] for p in self.draw_order]
        new.player_to_play = player_ids[id(self.player_to_play)]
        new.last_played_attacker = player_ids.get(id(self.last_played_attacker), None)
        new.defender = player_ids.get(id(self.defender), None)
        new.loser = player_ids.get(id(self.loser), None)

        ### Copy different, more general information
        new.is_end_state = self.is_end_state
        new.computer_shuffle = self.computer_shuffle
        new.all_cards = self.all_cards
        new.current_action = self.current_action
        new.current_attacker = self.current_attacker
        new.attacker_to_start_throwing = self.attacker_to_start_throwing
        new.reflected_trumps = self.reflected_trumps.copy()  # (suit, value) pairs
        new.history = self.history.copy()
        new.print_info = False

        ### And now we copy all the cards changing each card in all places
        new.card_collection = []
        card_ids = {}  # dict of all the cards with their copy
        for card in self.card_collection:
            # Change card on all different places
            copy_card = card.make_copy()
            card_ids[id(card)] = copy_card
            new.card_collection.append(copy_card)

        for player_idx, p in enumerate(self.players):
            new.players[player_idx].hand = [card_ids[id(c)] for c in p.hand]
        new.deck = [card_ids[id(c)] for c in self.deck]
        new.cards_to_defend = [card_ids[id(c)] for c in self.cards_to_defend]
        new.cards_killed = [(card_ids[id(p[0])], card_ids[id(p[1])]) for p in self.cards_killed]
        return new


    def next(self):
        """Chooses and performs an action/move"""
        # Check if this node is terminal
        if self.is_end_state:
            raise BaseException('This was an end state')

        # Print the current game state
        # if self.print_info:
            # print(basic_repr_game(self))

        # Choose an action
        action = self.player_to_play.choose_action(self)

        # Display the action
        if self.print_info or not self.computer_shuffle:
            if action[0] == 'Attack':
                # Check if action[1] is a list of tuples (multiple cards) or a single tuple
                if isinstance(action[1], list):
                    # Multiple cards case
                    card_strings = []
                    for card in action[1]:
                        card_str = "X♣♠♥♦"[card[0]] + "0123456789*JQKA"[card[1]]
                        card_strings.append(card_str)
                    cards_display = ", ".join(card_strings)
                    print(f'Action {action[0]} with cards {cards_display} was chosen by {self.player_to_play}')
                else:
                    # Single card case
                    card_str = "X♣♠♥♦"[action[1][0]] + "0123456789*JQKA"[action[1][1]]
                    print(f'Action {action[0]} with card {card_str} was chosen by {self.player_to_play}')
            elif action[0] in ['Defend', 'Reflect']:
                # NOTE: added +1 to suit to match the 1-4 range
                # TODO: Refactor cards printing to use tuples or CARD classes instead of strings
                print(
                    f'Action {action[0]} with card {"X♣♠♥♦"[action[1][0]] + "0123456789*JQKA"[action[1][1]]} was chosen by {self.player_to_play}')
            elif action[0] == 'ThrowCards':
                print(f'Action {action[0]} with {action[1]} was chosen by {self.player_to_play}')
            else:
                print(f'Action {action} was chosen by {self.player_to_play}')

        # Execute the action
        self.execute_action(action)


if __name__ == '__main__':
    import random
    random.seed(2)
    # Note the main attacker should be specified

    # The players can be one of ISMCTS, ISMCTSFPV, DeterminizedMCTS, Random, Human
    players = [Human('Player1'), Human('Player2')]

    # If the computer must shuffle the deck of cards instead the player in real-life
    # computer_shuffle = False
    computer_shuffle = True

    game = MoskaGame(players, computer_shuffle, main_attacker='Player1')
    while not game.is_end_state:
        game.next()

    print()
    print(f'Game is lost by {game.loser}')
    # print([str(card) for card in game.card_collection])