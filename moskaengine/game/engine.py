import random
import itertools
from collections import defaultdict

# Moskaengine imports
from ..game.deck import Card, StandardDeck
from ..players.human_player import HumanPlayer
from ..utils import basic_repr_game, game_action_repr, state_as_vector, save_game_vector

class MoskaGame:
    """
    The class for representing a game of Moska.
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
                 shuffle_deck = True,
                 save_vectors = False,
                 state_folder = "game_vectors",
                 file_format = "csv",
                 save_history = True,
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
        self.deck = StandardDeck(shuffle=shuffle_deck, perfect_info=perfect_info)
        self.card_collection = self.deck.copy()

        # Initialize
        self.all_cards_tuples = {(suit, value) for suit in range(1, 5) for value in range(2, 15)}
        self.all_cards = [Card(value, suit) for suit in range(1, 5) for value in range(2, 15)]

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
            self.deck = player.fill_hand(self.deck)
            # player.hand = self.deck.pop(6)

        if computer_shuffle:
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

                        if self.print_info:
                            print(f"{player} has switched the trump card ({self.trump_card}) with their 2 of trump card ({card})")

                        # Make both cards public
                        card.is_public = True
                        card.is_unknown = False
                        card.is_private = False
                        self.trump_card.is_public = True
                        self.trump_card.is_unknown = False
                        self.trump_card.is_private = False
                        self.trump_card.is_drawn = True

                        # Store the previous trump card to update it in the collection
                        previous_trump = self.trump_card
                        self.trump_card = card

                        # Update the drawn card in the card collection
                        for c in self.card_collection:
                            if c.suit == card.suit and c.value == card.value:
                                c.is_public = True
                                c.is_unknown = False
                                c.is_private = False
                                c.is_drawn = True
                                break

                        # Update the previous trump card in the card collection
                        for c in self.card_collection:
                            if c.suit == previous_trump.suit and c.value == previous_trump.value:
                                c.is_public = True
                                c.is_unknown = False
                                c.is_private = False
                                c.is_drawn = False
                                break
            else:
                # If no player has the 2 of trump, the card is placed at the bottom of the deck
                for c in self.card_collection:
                    if c.suit == self.trump_card.suit and c.value == self.trump_card.value:
                        c.is_public = True
                        c.is_unknown = False
                        c.is_private = False
                        c.is_drawn = False
                        break
                # Place the trump card at the bottom of the deck
                self.deck.place_bottom(self.trump_card)
        else:
            # TODO: Not implemented correctly yet
            # We must shuffle the cards irl
            print("Specify the suit and value of the trump card:")
            self.deck[-1].from_input(self.all_cards_tuples)
            self.deck[-1].is_public = True

        # Initialize the players
        # for player in self.players:
        #     # The suit and value of the card is not necessary at this point
        #     # only if the player is not Human the suit and value of the card
        #     # must be added whenever all possible actions are listed.
        #     self.deck = player.fill_hand(self.deck)

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

    def make_discarded_cards_public(self, cards, drawn=False):
        """Make the discarded cards public."""
        for card in cards:
            card.is_public = True
            card.is_unknown = False
            card.is_private = False
            card.is_drawn = drawn

        # Update the card collection
        for c in self.card_collection:
            if c in cards:
                c.is_public = True
                c.is_unknown = False
                c.is_private = False
                c.is_drawn = drawn

        for c in self.deck:
            if c in cards:
                c.is_public = True
                c.is_unknown = False
                c.is_private = False
                c.is_drawn = drawn

    def get_unknown_cards(self):
        """Returns the (suit, value) pairs of all unknown cards"""
        # remove = set()
        # for card in self.card_collection:
        #     if not card.is_unknown:
        #         remove.add((card.suit, card.value))
        remove = {(c.suit, c.value) for c in self.card_collection if not c.is_unknown}
        return self.all_cards_tuples - remove

    # def get_unknown_cards(self):
    #     """Returns the Card objects of all unknown cards"""
    #     unknown_cards = {c for c in self.card_collection if c.is_unknown}
    #     return unknown_cards

    def get_non_public_cards_tuples(self):
        """Returns the (suit, value) pairs of all cards that are not public"""
        # Using set comprehension for efficiency
        public_cards = {(c.suit, c.value) for c in self.card_collection if c.is_public}
        return self.all_cards_tuples - public_cards

    def get_non_public_cards(self):
        """Returns all Card objects that are not public."""
        return [c for c in self.card_collection if not c.is_public]

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
                # There is a pair on the table, the player can skip on attacking
                possible_actions.append(('Skip', None))

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

            # If there are cards to throw
            if max_throws > 0:
                for throw in range(1, max_throws + 1):
                    for option in itertools.combinations(possible_throws, r=throw):
                        if player.can_throw(self.get_non_public_cards_tuples(), list(option)):
                            if len(option) == 1:
                                possible_actions.append(('ThrowCards', option[0]))
                            else:
                                possible_actions.append(('ThrowCards', option))

            # Can always skip
            possible_actions.append(('Skip', None))
        else:
            raise ValueError(f"Unknown action {action}")

        if not possible_actions:
            # TODO: Bug may be when player has killed all cards during mcts but the killed_cards are empty -> turn does not transfer over to next player
            print(self.current_attacker, self.player_to_play, self.current_action, self.defender, self.cards_to_defend, self.cards_killed)
            print(f"Defender {self.defender} hand size: {len(self.defender.hand)} and cards_by_value: {cards_by_value}")
            print(f"No possible actions for {player.name}, game state: {basic_repr_game(self)}")
            print(f"Players remaining: {[p.name for p in self.attackers]}")
            if self.current_action == 'Attack' and len(self.defender.hand) == 0 and len(self.deck) == 0:
                # If the defender has no cards left, they are out of the game and the next attacker is the one to the left of the defender
                self.new_attack(self.attackers[(self.current_attacker + 1) % len(self.attackers)])
            else:
                raise BaseException(f"No possible actions for {player.name} in {action}")

        return possible_actions
    

    def execute_action(self, action):
        # Add to history
        self.history.append((action, self.player_to_play.name))
        if len(self.history) > self.N_HISTORY:
            self.history.pop(0)
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
                # TODO: This is a bug when the defender defends all and the deck is empty, they should be out of the game. This is does not happen in the current game logic.
                # No more cards to defend, switch to attacking
                if len(self.defender.hand) == 0 and len(self.deck) == 0:
                    # print(f"Defender {self.defender} has no cards left, switching to the next attacker.")
                    # print(f"Cards killed: {self.cards_killed}")
                    # print(f"Cards to defend: {self.cards_to_defend}")
                    # print(basic_repr_game(self))
                    # print(f"Players remaining: {[p.name for p in self.attackers]}")
                    # print(self.player_to_play, self.attackers, self.current_attacker, self.current_action)
                    # If the defender has no cards left, they are out of the game and the next attacker is the one to the left of the defender
                    #self.attackers.remove(self.defender)
                    self.new_attack(self.attackers[(self.current_attacker + 1) % len(self.attackers)])
                    # self.defender = self.attackers[self.current_attacker % len(self.attackers)]
                    # self.player_to_play = self.attackers[self.current_attacker]
                    # self.current_action = 'Attack'
                else:
                    self.player_to_play = self.attackers[self.current_attacker]
                    self.current_action = 'Attack'

            # Make the played cards public
            self.make_discarded_cards_public(cards_played + cards_defended)
            return

        elif action_type == 'PlayFromDeck':
            # Draw a card to play from the deck i.e., Koplaus
            kopled_card = self.deck.pop(1)[0]
            kopled_card.kopled = kopled_card.is_public = True
            kopled_card.is_unknown = kopled_card.is_private = False
            card_to_defend = action[1][1]

            if card_to_defend and card_to_defend in self.cards_to_defend:
                # Kopled card can defend a card on the table
                self.cards_to_defend.remove(card_to_defend)
                self.cards_killed.append((card_to_defend, kopled_card))
                self.cards_discarded.extend([kopled_card, card_to_defend])
                # Make the played cards public
                self.make_discarded_cards_public([kopled_card, card_to_defend])
            else:
                # If the kopled card can't defend a card on the table, it is added to the cards_to_defend
                self.cards_to_defend.append(kopled_card)
                # Make the kopled card public even if it can't defend
                self.make_discarded_cards_public([kopled_card])

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

        elif action_type == 'ThrowCards':
            # PlayToOther i.e., play to table for the defender to fall
            if len(action[1]) == 1:
                card_played = self.player_to_play.discard_card(self, action[1].suit, action[1].value)
                self.cards_to_defend.append(card_played)
            else:
                for card in action[1]:
                    card_played = self.player_to_play.discard_card(self, card.suit, card.value)
                    self.cards_to_defend.append(card_played)

            # Make the played cards public
            self.make_discarded_cards_public(self.cards_to_defend[-len(action[1]):])

            # Give the defender a new chance if cards changed
            self.last_played_attacker = self.player_to_play
            self.player_to_play = self.defender
            self.current_action = 'Defend'

        elif action_type == 'Skip':
            # Increment attacker and player to play
            self.current_attacker = (self.current_attacker + 1) % len(self.attackers)
            self.player_to_play = self.attackers[self.current_attacker]

            # If there are cards on the table, the defender must pick them up
            if self.cards_to_defend:
                # Check if everybody got a chance to throw
                if self.player_to_play == self.attackers[self.attacker_to_start_throwing]:
                    if self.draw_undefended:
                        # NOTE: The defender picks up the cards here.
                        # The defender takes the non-defended cards, and the new main attacker is the one to the left of the defender
                        self.defender.hand.extend(self.cards_to_defend)
                        self.draw_undefended = False
                        # Make the drawn cards public
                        self.make_discarded_cards_public(self.cards_to_defend, drawn=True)
                    else:
                        all_on_table = [card for pair in self.cards_killed for card in pair] + self.cards_to_defend
                        self.defender.hand.extend(all_on_table)
                        # Make the drawn cards public
                        self.make_discarded_cards_public(all_on_table, drawn=True)

                    for player in self.draw_order:
                        self.deck = player.fill_hand(self.deck)

                    # Defender takes the cards and the new main attacker is the one to the left of the defender
                    self.new_attack(self.attackers[1 % len(self.attackers)])
                return
            else:
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

    def clone_for_rollout(self):
        """Clone the game for a rollout."""
        players = self.players
        copy_card = Card.make_copy
        
        new = MoskaGame(None, False, None, False, False)

        # Copy players and build player_ids mapping
        player_ids = {id(pl): pl.make_copy() for pl in players}
        new.players = list(player_ids.values())
        new.player_ids = {player_ids[id(pl)]: idx for pl, idx in self.player_ids.items()}

        new.deck = StandardDeck(
            shuffle=False,
            perfect_info=self.perfect_info,
            cards=list(map(copy_card, self.deck.cards))
        )

        # Copy card lists
        new.cards_to_defend = list(map(copy_card, self.cards_to_defend))
        new.cards_killed = [(copy_card(card[0]), copy_card(card[1])) for card in self.cards_killed]
        new.cards_discarded = list(map(copy_card, self.cards_discarded))
        new.trump_card = copy_card(self.trump_card)
        new.card_collection = list(map(copy_card, self.card_collection))

        # Direct references for simple collections
        new.all_cards_tuples = self.all_cards_tuples
        new.all_cards = self.all_cards

        # Copy simple player references using player_ids mapping
        new.attackers = [player_ids[id(p)] for p in self.attackers]
        new.defender = player_ids[id(self.defender)] if self.defender else None
        new.draw_order = [player_ids[id(p)] for p in self.draw_order]
        new.player_to_play = player_ids[id(self.player_to_play)]
        new.last_played_attacker = player_ids[id(self.last_played_attacker)] if self.last_played_attacker else None
        new.loser = player_ids[id(self.loser)] if self.loser else None

        # Copy other attributes
        attrs_to_copy = [
            'current_attacker', 'current_action', 'draw_undefended', 'attacker_to_start_throwing', 'n_turns', 'perfect_info'
        ]
        for attr in attrs_to_copy:
            setattr(new, attr, getattr(self, attr))

        # Copy the history
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
            # Print the game state for computer players
            if not isinstance(self.player_to_play, HumanPlayer):
                print(basic_repr_game(self))
            print(game_action_repr(self.player_to_play, action))

        # Execute the action
        self.execute_action(action)
        self.n_turns += 1

        if self.is_end_state and self.save_vectors:
            # Save the game vector
            if self.debug:
                save_game_vector(self.state_data, self.opponent_data, self.state_folder, self.file_format)

            return self.state_data, self.opponent_data

        return None