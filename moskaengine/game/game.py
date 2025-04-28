from moskaengine.game.deck import Card, suit_to_symbol, StandardDeck
from moskaengine.utils.card_utils import choose_random
from moskaengine.players.human_player import Human
from moskaengine.utils.card_utils import basic_repr_game

from itertools import combinations
import random
from collections import defaultdict

from moskaengine.utils.game_utils import state_as_vector, save_game_vector


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

    # Number of turns to keep track of for vector representation
    N_HISTORY = 5

    def __init__(self,
                 players,
                 computer_shuffle,
                 main_attacker = None,
                 do_init = True,
                 print_info = False,
                 perfect_info = False,
                 save_vectors = False,
                 state_folder = "game_vectors",
                 file_format = "csv",
                 debug = False,
                 ):

        if not do_init:
            return

        self.players = players
        self.computer_shuffle = computer_shuffle
        self.print_info = print_info
        self.perfect_info = perfect_info
        self.n_turns = 0
        self.debug = debug

        # Initialize the player variables
        self.attackers = None
        self.defender = None
        self.current_attacker = None
        self.draw_undefended = False

        # Game variables
        self.cards_killed = []
        self.cards_to_defend = []

        # Data saving
        self.save_vectors = save_vectors
        self.state_folder = state_folder
        self.file_format = file_format
        self.history = []
        self.state_data = []
        self.opponent_data = []

        # If the main attacker is not specified, choose randomly
        self.main_attacker = random.choice(players) if main_attacker is None else next(player for player in players if player.name == main_attacker)

        # NOTE: Only initialized for the game state vector
        self.player_to_play = self.main_attacker

        # Initialize the deck
        self.deck = StandardDeck(shuffle=True, perfect_info=perfect_info)

        # Initialize
        self.all_cards = {(suit, value) for suit in range(1, 5) for value in range(2, 15)}

        self.card_collection = self.deck.copy()

        self.reference_deck = StandardDeck(shuffle=False, perfect_info=True)

        # Players draw their hand
        for player in self.players:
            player.hand = self.deck.pop(6)

        if computer_shuffle:
            unknown = self.get_unknown_cards()

            # Draw trump card
            self.trump_card = self.deck.pop(1)[0]
            self.trump_card.is_public = True
            # print(f"Trump card: {self.trump_card}")

            # Check if a player has the equivalent of 2 of trump and if so, switch the card
            for player in self.players:
                # Find the 2 of trump card in players' hand (if it exists)
                for card in player.hand:
                    if card.suit == self.trump_card.suit and card.value == 2:
                        player.hand.remove(card)
                        player.hand.append(self.trump_card)
                        self.deck.place_bottom(card)

                        if self.print_info:
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

        # # TODO: This is a weird way to check trump
        # for card in self.deck.cards:
        #     card.trump_suit = self.trump_card.suit

        # Initialize the players
        for player in self.players:
            # The suit and value of the card is not necessary at this point
            # only if the player is not Human the suit and value of the card
            # must be added whenever all possible actions are listed.
            self.deck = player.fill_hand(self.deck)


        # Initialize the attacking player
        if self.print_info:
            print(f"Player to start: {self.main_attacker}")

        # Game state vector for the initial game
        if self.save_vectors:
            state, opponent = state_as_vector(self)
            self.state_data.append(state)
            self.opponent_data.append(opponent)

        # Start the game
        self.new_attack(self.main_attacker)

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
            if player == main_attacker:
                break
        else:
            raise BaseException("Starting player not found.")

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

        # The action to perform
        self.current_action = 'Attack'
        # Succesfully defended cards as (attack, defend) pairs
        self.cards_killed = []
        self.cards_to_defend = []

        # TODO: Initialize game vector here

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
                possible_plays = [card for card in possible_plays if card.value in values_on_table]

            # Group cards by value
            cards_by_value = {}
            for card in possible_plays:
                if card.value not in cards_by_value:
                    cards_by_value[card.value] = []
                cards_by_value[card.value].append(card)

            defender_hand_size = len(self.defender.hand)
            current_attack_count = len(self.cards_to_defend)
            max_additional_attacks = defender_hand_size - current_attack_count

            # Check if table has space to play new cards for the defender
            if max_additional_attacks > 0:
                # Single card attacks
                for card in possible_plays:
                    possible_actions.append(('Attack', card))

                # A combination card attack
                for value, cards in cards_by_value.items():
                    if len(cards) > 1:
                        import itertools

                        for i in range(2, min(len(cards) + 1, max_additional_attacks + 1)):
                            for combination in itertools.combinations(cards, i):
                                if len(combination) <= max_additional_attacks:
                                    possible_actions.append(('Attack', combination))

        elif action == "Defend":
            # Iterate through the cards in the hand to see which ones can be played
            for to_defend in self.cards_to_defend:
                defend_options = []

                for card in player.hand:
                    # Check if the card can defend the to_defend card
                    if card.suit == self.trump_card.suit and not to_defend.is_trump():
                        # Can always defend with trump on a non-trump card
                        defend_options.append(('Defend', (card, to_defend)))
                    elif card.suit == to_defend.suit and card.value > to_defend.value:
                        # Can always defend the same suit with higher value
                        defend_options.append(('Defend', (card, to_defend)))

                # Add defense options for this card with appropriate weights
                if defend_options:
                    weight = 1
                    for option in defend_options:
                        possible_actions.append(option + (weight,))

            ## Koplaus
            # Check if there is already a kopled card on the table
            has_kopled_card = any(card.kopled for card in self.cards_to_defend)

            if not has_kopled_card and len(self.deck) > 0:
                next_card = self.deck.cards[0]
                next_card.is_public = True
                next_card.is_unknown = False

                for defend_card in self.cards_to_defend:
                    # Check if the kopled card could fall a card on the table
                    if next_card.suit == defend_card.suit and next_card.value > defend_card.value:
                        possible_actions.append(('PlayFromDeck', (next_card, defend_card), 1 / 2, True))
                    elif next_card.suit == self.trump_card.suit and not defend_card.is_trump():
                        possible_actions.append(('PlayFromDeck', (next_card, defend_card), 1 / 2, True))
                    else:
                        # If the kopled card can't fall a card on the table, it must be added to the cards_to_defend
                        possible_actions.append(('PlayFromDeck', (next_card, None), 1 / 2, False))

            # You can also pick up only the cards on the table if there are defended cards
            if self.cards_killed:
                possible_actions.append(('TakeDefend', None, 1 / (len(self.cards_to_defend)) + 2))

            # As the defender you can also always pick up all the cards
            possible_actions.append(('TakeAll', None, 1 / (len(self.cards_to_defend + self.cards_killed) + 2)))


        elif action == "ThrowCards":
            # Used when there are cards still on the table, otherwise attack action is used.

            possible_throws = player.possible_card_plays(self.get_non_public_cards())

            # You can only throw cards with the same value as those on the table
            values_on_table = {card.value for pair in self.cards_killed for card in pair}

            values_on_table.update({card.value for card in self.cards_to_defend})

            possible_throws = {card for card in possible_throws if card in values_on_table}
            available_throws = len(self.defender.hand) - len(self.cards_to_defend)

            # If 0 cards thrown
            possible_actions.append(('ThrowCards', None))

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

        # print(f"Found possible actions: {possible_actions}")
        return possible_actions

    def get_id(self):
        return hash(tuple(self.history))

    def execute_action(self, action):
        # Add to history
        self.history.append((action, self.player_to_play.name))

        if action[0] == 'Attack':
            # The attacking move i.e. the first play to an empty table
            if len(action[1]) > 1:
                for card in action[1]:
                    card_played = self.player_to_play.discard_card(self, card.suit, card.value)
                    self.cards_to_defend.append(card_played)
            else:
                card = action[1]
                card_played = self.player_to_play.discard_card(self, card.suit, card.value)
                self.cards_to_defend.append(card_played)

            # The attacking player draws card from the deck until their hand full after attacking
            self.player_to_play.fill_hand(self.deck)

            # Update values
            self.last_played_attacker = self.player_to_play
            self.player_to_play = self.defender
            self.current_action = 'Defend'

        elif action[0] == 'Defend':
            # TODO: Improve performance with sets?
            # The defending move from the target player for the current turn
            cards_played = []
            cards_defended = []

            # print(action, action[1], len(action[1]))

            # Check if multiple cards are defended at the same time
            if isinstance(action[1], tuple):
                # Single case
                played_card = self.player_to_play.discard_card(self, action[1][0].suit, action[1][0].value)
                cards_played.append(played_card)
                self.cards_to_defend.remove(action[1][1])
                cards_defended.append(action[1][1])
            elif isinstance(action[1], list):
                # Multiple case
                for played_card in action[1]:
                    suit, value = played_card.suit, played_card.value
                    card_played = self.player_to_play.discard_card(self, suit, value)
                    cards_played.append(card_played)

                for defended_card in action[2]:
                    self.cards_to_defend.remove(defended_card)
                    cards_defended.append(defended_card)
            else:
                raise ValueError(f"Action[1] is not a tuple or list: {action[1]}")

            for i in range(len(cards_defended)):
                if i < len(cards_played):
                    self.cards_killed.append((cards_defended[i], cards_played[i]))

            if len(self.cards_to_defend) == 0:
                # No more cards to defend, switch to attacking
                self.player_to_play = self.attackers[self.current_attacker
                ]
                self.current_action = 'Attack'

        elif action[0] == 'PlayFromDeck':
            # Draw a card from the deck i.e. Koplaus
            kopled_card = self.deck.pop(1)[0]
            kopled_card.kopled = True

            card_to_defend = action[1][1]

            # If this is False, the card can't fall a card on the table, and therefore it is added to the cards_to_defend

            if not action[2]:
                self.cards_to_defend.append(kopled_card)

            # Otherwise, the card can fall a card on the table
            else:
                if card_to_defend in self.cards_to_defend:
                    self.cards_to_defend.remove(card_to_defend)
                    card_defended = card_to_defend
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

            if action[1] is not None:

                cards_to_throw = action[1]

                for suit, value in cards_to_throw:
                    card_played = self.player_to_play.discard_card(self, suit, value)
                    self.cards_to_defend.append(card_played)

                # Give the defender a new chance if cards changed
                self.last_played_attacker = self.player_to_play
                self.player_to_play = self.defender
                self.current_action = 'Defend'
            else:
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
                self.new_attack(self.attackers[1 % len(self.attackers)])

        elif action[0] == 'PassAttack':
            # Pass on attacking i.e. skip
            self.current_attacker = (self.current_attacker + 1) % len(self.attackers)
            self.player_to_play = self.attackers[self.current_attacker]

            # Check if no one wants to attack
            if self.player_to_play == self.last_played_attacker:
                # The defender defended successfully
                for player in self.draw_order:

                    self.deck = player.fill_hand(self.deck)

                assert self.cards_to_defend == []

                self.new_attack(self.defender)
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

        # Game state as a vector for each turn before the next action
        if self.n_turns >= 1 and self.save_vectors:
            state, opponent = state_as_vector(self)
            self.state_data.append(state)
            self.opponent_data.append(opponent)

        # Choose an action
        action = self.player_to_play.choose_action(self)

        # Display the action
        if self.print_info or not self.computer_shuffle:
            if action[0] in ['Attack', 'Defend']:
                # Check if action[1] is a list of tuples (multiple cards) or a single tuple
                if len(action[1]) > 1:
                    print(f"{self.player_to_play} plays {action[0]} with cards {action[1]}\n")
                else:
                    print(f"{self.player_to_play} plays {action[0]} with card {action[1]}\n")
            elif action[0] == 'Defend':
                print(action, len(action[1]))
                for i in enumerate(action[1][0]):
                    print(f"{self.player_to_play} defends card {action[1][0][i]} with {action[1][1][i]}\n")

                # print(
                #     f'Action {action[0]} with card {"X♣♠♥♦"[action[1][0].suit] + "0123456789*JQKA"[action[1][1]]} was chosen by {self.player_to_play}')
            elif action[0] == 'ThrowCards':
                print(f'Action {action[0]} with {action[1]} was chosen by {self.player_to_play}\n')
            elif action[0] == 'PlayFromDeck':
                print(f"Action {action[0]} was chosen by {self.player_to_play} and the drawn card was {action[1][0]}\n")
            else:
                print(f'Action {action} was chosen by {self.player_to_play}\n')

        # Execute the action
        self.execute_action(action)
        self.n_turns += 1

        if self.is_end_state and self.save_vectors:
            # Save the game vector
            if self.debug:
                save_game_vector(self.state_data, self.opponent_data, self.state_folder, self.file_format)

            return self.state_data, self.opponent_data

        return None