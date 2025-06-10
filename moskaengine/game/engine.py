from moskaengine.game.deck import Card, suit_to_symbol, StandardDeck
from moskaengine.utils.card_utils import choose_random
from moskaengine.players.human_player import Human
from moskaengine.utils.card_utils import basic_repr_game

import itertools
import random
from collections import deque, defaultdict

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
                 print_info = True,
                 perfect_info = False,
                 save_vectors = False,
                 state_folder = "game_vectors",
                 file_format = "csv",
                 debug = False,
                 ):

        if not do_init:
            return

        self.players = players
        self.player_ids = {player: idx for idx, player in enumerate(players, 1)}
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
        self.last_player_attacker = None
        self.current_action = None
        self.attacker_to_start_throwing = None
        self.last_played_attacker = None

        # Initialize the deck
        self.reference_deck = StandardDeck(shuffle=False, perfect_info=True)
        self.deck = StandardDeck(shuffle=True, perfect_info=perfect_info)
        self.card_collection = [card.make_copy() for card in self.deck.cards]

        # Initialize
        self.all_cards = {(suit, value) for suit in range(1, 5) for value in range(2, 15)}

        # Game variables
        self.cards_killed = []
        self.cards_to_defend = []
        self.cards_discarded = []

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



        # Players draw their hand
        for player in self.players:
            player.hand = self.deck.pop(6)

        if computer_shuffle:
            unknown = self.get_unknown_cards()

            # Draw trump card
            self.trump_card = self.deck.pop(1)[0]
            self.trump_card.is_public = True
            self.trump_card.is_unknown = False
            self.trump_card.is_private = False
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
                        card.is_private = False
                        self.trump_card.is_public = True
                        self.trump_card.is_unknown = False
                        self.trump_card.is_private = False

                        self.trump_card = card
                        break
        else:
            # TODO: Not implemented correctly yet
            # We must shuffle the cards irl
            print("Specify the suit and value of the trump card:")
            self.deck[-1].from_input(self.all_cards)
            self.deck[-1].is_public = True

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
        remove = {(c.suit, c.value) for c in self.card_collection if not c.is_unknown}
        return self.all_cards - remove

    # def get_unknown_cards(self):
    #     """Returns the Card objects of all unknown cards"""
    #     unknown_cards = {c for c in self.card_collection if c.is_unknown}
    #     return unknown_cards

    def get_non_public_cards_tuples(self):
        """Returns the (suit, value) pairs of all unknown cards + private cards"""
        # remove = set()
        # for card in self.card_collection:
        #     if card.is_public:
        #         remove.add((card.suit, card.value))
        remove = {(c.suit, c.value) for c in self.card_collection if c.is_public}
        return self.all_cards - remove

    def get_non_public_cards(self):
        """Returns all Card objects that are not public."""
        return [card for card in self.card_collection if not card.is_public]

    # def get_non_public_cards(self):
    #     """Returns the Card objects of all non-public cards"""
    #     remove = {(c.suit, ) for c in self.card_collection if c.is_public}
    #     return self.all_cards - remove

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

    def get_id(self):
        return hash(tuple(self.history))

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
            if self.cards_killed:
                # There is a pair on the table, the player can pass on attacking
                possible_actions.append(('PassAttack', None))

                # If no pass, we play cards with the same values as those that are played to table
                values_on_table = {card.value for pair in self.cards_killed for card in pair}
                possible_plays = [card for card in possible_plays if card.value in values_on_table]

            # Group cards by value
            cards_by_value = defaultdict(list)
            for card in possible_plays:
                cards_by_value[card.value].append(card)

            defender_hand_size = len(self.defender.hand)
            max_additional_attacks = defender_hand_size - len(self.cards_to_defend)

            # Check if the table has space to play new cards for the defender
            if max_additional_attacks > 0:
                # Single card attacks
                possible_actions.extend(('Attack', card) for card in possible_plays)

                # A combination card attack
                for cards in cards_by_value.values():
                    if len(cards) > 1:
                        max_combinations = min(len(cards), max_additional_attacks)
                        for i in range(2, max_combinations + 1):
                            for combination in itertools.combinations(cards, i):
                                possible_actions.append(('Attack', combination))

        elif action == "Defend":
            # Iterate through the cards in the hand to see which ones can be played
            for to_defend in self.cards_to_defend:
                for card in player.hand:
                    # Check if the card can defend the to_defend card
                    if (card.suit == self.trump_card.suit and to_defend.suit != self.trump_card.suit) or (card.suit == to_defend.suit and card.value > to_defend.value):
                        # Can defend with a trump on a non-trump card or with a greater value card
                        possible_actions.append(('Defend', (card, to_defend), 1))

            # Koplaus logic
            # Check if there is already a kopled card on the table
            if self.deck and not any(card.kopled for card in self.cards_to_defend):
                next_card = self.deck.cards[0]
                # next_card.is_private = True
                # next_card.is_unknown = False

                for defend_card in self.cards_to_defend:
                    # Check if the kopled card could fall a card on the table
                    if (next_card.suit == defend_card.suit and next_card.value > defend_card.value) or (next_card.suit == self.trump_card.suit and to_defend.suit != self.trump_card.suit):
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
            values_on_table.update(card.value for card in self.cards_to_defend)

            possible_throws = [card for card in possible_throws if card.value in values_on_table]

            available_throws = len(self.defender.hand) - len(self.cards_to_defend)
            max_throws = min(available_throws, len(possible_throws), len(player.hand))

            # Default case
            possible_actions.append(('ThrowCards', None))

            # If there are cards to throw
            if max_throws > 0:
                for throw in range(1, max_throws + 1):
                    for option in itertools.combinations(possible_throws, r=throw):
                        if player.can_throw(self.get_non_public_cards_tuples(), list(option)):
                            if len(option) == 1:
                                possible_actions.append(('ThrowCards', option[0]))
                            else:
                                possible_actions.append(('ThrowCards', option))

        else:
            raise ValueError(f"Unknown action {action}")

        if not possible_actions:
            raise BaseException(f"No possible actions for {player.name} in {action}")

        return possible_actions


    def execute_action(self, action):
        # Add to history
        self.history.append((action, self.player_to_play.name))
        action_type = action[0]

        if action_type == 'Attack':
            cards = action[1] if isinstance(action[1], tuple) else [action[1]]

            for card in cards:
                card_played = self.player_to_play.discard_card(self, card.suit, card.value)
                self.cards_to_defend.append(card_played)

            # The attacking player draws cards from the deck until their hand is full after attacking
            self.player_to_play.fill_hand(self.deck)

            # Update values
            self.last_played_attacker = self.player_to_play
            self.player_to_play = self.defender
            self.current_action = 'Defend'
            return

        elif action_type == 'Defend':
            # TODO: Improve performance with sets?
            # The defending move from the target player for the current turn
            cards_played, cards_defended = [], []
            # Check if multiple cards are defended at the same time
            if isinstance(action[1], tuple):
                # Single case
                played_card = self.player_to_play.discard_card(self, action[1][0].suit, action[1][0].value)
                self.cards_to_defend.remove(action[1][1])
                cards_played.append(played_card)
                cards_defended.append(action[1][1])
            elif isinstance(action[1], list):
                # Multiple case
                for played_card in action[1]:
                    card_played = self.player_to_play.discard_card(self, played_card.suit, played_card.value)
                    cards_played.append(card_played)
                for defended_card in action[2]:
                    self.cards_to_defend.remove(defended_card)
                    cards_defended.append(defended_card)
            else:
                raise ValueError(f"Action[1] is not a tuple or list: {action[1]}")

            for defended_card, played_card in zip(cards_defended, cards_played):
                self.cards_killed.append((defended_card, played_card))
                self.cards_discarded.extend([defended_card, played_card])

            if not self.cards_to_defend:
                # No more cards to defend, switch to attacking
                self.player_to_play = self.attackers[self.current_attacker]
                self.current_action = 'Attack'
            return

        elif action_type == 'PlayFromDeck':
            # Draw a card to play from the deck i.e., Koplaus
            kopled_card = self.deck.pop(1)[0]
            kopled_card.kopled = kopled_card.is_public = True
            kopled_card.is_unknown = kopled_card.is_private = False
            card_to_defend = action[1][1]

            # If the kopled card can't defend a card on the table, it must be added to the cards_to_defend
            if card_to_defend and card_to_defend in self.cards_to_defend:
                self.cards_to_defend.remove(card_to_defend)
                self.cards_killed.append((card_to_defend, kopled_card))
                self.cards_discarded.extend([kopled_card, card_to_defend])
            else:
                self.cards_to_defend.append(kopled_card)

            # If all the cards are defended, the game continues accordingly
            if not self.cards_to_defend:
                # No more cards to defend, switch to attacking
                self.player_to_play = self.attackers[self.current_attacker]
                self.current_action = 'Attack'
            return

        elif action_type in ('TakeAll', 'TakeDefend'):
            # Take cards from the table, should be only available for the defender
            assert self.player_to_play == self.defender
            if action_type == 'TakeDefend':
                self.draw_undefended = True
            self.current_action = 'ThrowCards'
            self.player_to_play = self.attackers[self.current_attacker]
            self.attacker_to_start_throwing = self.current_attacker
            return

        if action_type == 'ThrowCards':
            # PlayToOther i.e., play to table for the defender to fall
            if action[1] is not None:
                if len(action[1]) == 1:
                    card_played = self.player_to_play.discard_card(self, action[1].suit, action[1].value)
                    self.cards_to_defend.append(card_played)
                else:
                    for card in action[1]:
                        card_played = self.player_to_play.discard_card(self, card.suit, card.value)
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
                    # The defender takes the non-defended cards, and the new main attacker is the one to the left of the defender
                    self.defender.hand.extend(self.cards_to_defend)
                    self.draw_undefended = False
                else:
                    all_on_table = [card for pair in self.cards_killed for card in pair] + self.cards_to_defend
                    self.defender.hand.extend(all_on_table)

                for player in self.draw_order:
                    self.deck = player.fill_hand(self.deck)

                # Defender takes the cards and the new main attacker is the one to the left of the defender
                self.new_attack(self.attackers[1 % len(self.attackers)])
            return

        elif action_type == 'PassAttack':
            # Pass on attacking i.e. skip
            self.current_attacker = (self.current_attacker + 1) % len(self.attackers)
            self.player_to_play = self.attackers[self.current_attacker]

            # Check if no one wants to attack
            if self.player_to_play == self.last_played_attacker:
                # The defender defended successfully
                for player in self.draw_order:
                    self.deck = player.fill_hand(self.deck)

                assert not self.cards_to_defend
                self.new_attack(self.defender)
            return
        else:
            raise NotImplementedError('Action to execute not implemented.')

    # def clone_for_rollout(self):
    #     new = MoskaGame(None, False, None, False, False)
    #
    #     # Copy players and related attributes
    #     new.players = []
    #     player_ids = {}
    #     for pl in self.players:
    #         copy_pl = pl.make_copy()
    #         player_ids[id(pl)] = copy_pl
    #         new.players.append(copy_pl)
    #
    #     card_ids = {}
    #     # Helper function to copy a list of cards and update card_ids
    #     def get_copied_card(card):
    #         if card is None:
    #             return None
    #         card_id = id(card)
    #         if card_id not in card_ids:
    #             copy_card = card.make_copy()
    #             card_ids[card_id] = copy_card
    #             return copy_card
    #         return card_ids[card_id]
    #
    #
    #     new.attackers = [player_ids[id(p)] for p in self.attackers]
    #     new.defender = player_ids.get(id(self.defender), None)
    #     new.current_attacker = self.current_attacker
    #     new.current_action = self.current_action
    #     new.draw_undefended = self.draw_undefended
    #     new.draw_order = [player_ids[id(p)] for p in self.draw_order]
    #     new.player_to_play = player_ids[id(self.player_to_play)]
    #     new.attacker_to_start_throwing = self.attacker_to_start_throwing
    #     new.last_played_attacker = player_ids.get(id(self.last_played_attacker), None)
    #     new.loser = player_ids.get(id(self.loser), None)
    #
    #     # General game attributes
    #     new.computer_shuffle = self.computer_shuffle
    #     new.print_info = False
    #     new.perfect_info = self.perfect_info
    #     new.n_turns = self.n_turns
    #     new.debug = False
    #     new.is_end_state = self.is_end_state
    #
    #     # Data saving attributes
    #     new.history = self.history.copy()
    #     new.state_data = self.state_data.copy()
    #     new.opponent_data = self.opponent_data.copy()
    #     new.save_vectors = self.save_vectors
    #     new.state_folder = self.state_folder
    #     new.file_format = self.file_format
    #
    #     # Copy deck and card collection
    #     new_deck_cards = [get_copied_card(card) for card in self.deck.cards]
    #     new.deck = StandardDeck(shuffle=False, perfect_info=self.perfect_info)
    #     new.deck.cards = deque(new_deck_cards)
    #     new.card_collection = [get_copied_card(card) for card in self.card_collection]
    #     new.reference_deck = self.reference_deck
    #     new.trump_card = get_copied_card(self.trump_card)
    #     new.all_cards = self.all_cards  # Assuming no direct mutable Card objects
    #
    #     # Copy lists of cards using get_copied_card
    #     new.cards_killed = [(get_copied_card(p[0]), get_copied_card(p[1])) for p in self.cards_killed]
    #     new.cards_to_defend = [get_copied_card(card) for card in self.cards_to_defend]
    #     new.cards_discarded = [get_copied_card(card) for card in self.cards_discarded]
    #
    #     # Copy players' hands
    #     for idx, player in enumerate(self.players):
    #         new.players[idx].hand = [get_copied_card(c) for c in player.hand]
    #
    #     return new

    def clone_for_rollout(self):
        new = MoskaGame(None, False, None, False, False)

        # Copy players and related attributes
        new.players = []
        player_ids = {}
        for pl in self.players:
            copy_pl = pl.make_copy()
            player_ids[id(pl)] = copy_pl
            new.players.append(copy_pl)

        # Copy player_ids mapping from old players to new players
        new.player_ids = {player_ids[id(pl)]: idx for pl, idx in self.player_ids.items()}

        new.deck = StandardDeck(shuffle=False, perfect_info=self.perfect_info)
        new.deck.cards = deque(card.make_copy() for card in self.deck.cards)
        new.cards_to_defend = [card.make_copy() for card in self.cards_to_defend]
        new.cards_killed = [(card[0].make_copy(), card[1].make_copy()) for card in self.cards_killed]
        new.cards_discarded = [card.make_copy() for card in self.cards_discarded]
        new.trump_card = self.trump_card.make_copy()
        new.card_collection = self.card_collection
        new.all_cards = self.all_cards

        new.attackers = [player_ids[id(p)] for p in self.attackers]
        new.defender = player_ids.get(id(self.defender), None)
        new.current_attacker = self.current_attacker
        new.current_action = self.current_action
        new.draw_undefended = self.draw_undefended
        new.draw_order = [player_ids[id(p)] for p in self.draw_order]
        new.player_to_play = player_ids[id(self.player_to_play)]
        new.attacker_to_start_throwing = self.attacker_to_start_throwing
        new.last_played_attacker = player_ids.get(id(self.last_played_attacker), None)
        new.loser = player_ids.get(id(self.loser), None)

        new.n_turns = self.n_turns
        new.perfect_info = self.perfect_info

        # History
        new.history = self.history.copy()

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
            action_type = action[0]
            if action_type == 'Attack' or action_type == 'Defend':
                # Check if action[1] is a list of tuples (multiple cards) or a single tuple
                if len(action[1]) > 1:
                    print(f"{self.player_to_play} plays {action_type} with cards {action[1]}\n")
                else:
                    print(f"{self.player_to_play} plays {action_type} with card {action[1]}\n")
            # elif action_type == 'Defend':
            #     for i in enumerate(action[1][0]):
            #         print(f"{self.player_to_play} defends card {action[1][0][i]} with {action[1][1][i]}\n")
            elif action_type == 'ThrowCards':
                print(f'Action {action_type} with {action[1]} was chosen by {self.player_to_play}\n')
            elif action_type == 'PlayFromDeck':
                print(f"Action {action_type} was chosen by {self.player_to_play} and the drawn card was {action[1][0]}\n")
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