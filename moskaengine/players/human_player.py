from moskaengine.players.abstract_player import AbstractPlayer
from moskaengine.utils.card_utils import basic_repr_game, basic_repr_player_actions
from moskaengine.utils.player_utils import has_combinations


class Human(AbstractPlayer):
    def make_copy(self):
        new = Human(self.name)
        new.hand = self.hand.copy()
        return new

    def choose_action(self, game_state):
        if game_state.computer_shuffle:
            # The computer shuffles, so we can choose what to do
            # from checking the cards in our hand
            self.make_cards_known(game_state)

            # Now all cards are known, check the allowed plays
            allowed_actions = game_state.allowed_plays()
        else:
            # We must choose an allowed action, however since humans (i.e. real life
            # players) do not need to know their cards (if they shuffle themselves)
            # the allowed plays must return all possible actions with all possible
            # cards this player can have (in that case).
            allowed_actions = game_state.allowed_plays()

        ### Choose action from allowed actions
        action_types = {action[0] for action in allowed_actions}
        # Print the game state
        print(basic_repr_game(game_state))

        # Print the allowed actions
        print(basic_repr_player_actions(action_types, self))

        # Print the allowed actions
        # print()
        # print([c for c in self.hand])
        # print(f'Allowed actions for {self} are:')
        # print([i[0] for i in allowed_actions])
        # print([(i[0], i[1]) for i in allowed_actions])
        print(f'What does {self} do?')

        if len(action_types) > 1:
            action_types = sorted(action_types)

            # Ask the user to choose an action
            while True:
                try:
                    # Subtract 1 from the input to get the correct index
                    idx = int(input(f'Choose action (1 - {len(action_types)}): ')) - 1
                    # Check if the index is valid
                    if idx in range(0, len(action_types)):
                        break
                    else:
                        print(f'Index not valid, try again.\n')
                        print(basic_repr_player_actions(action_types, self))
                except (ValueError, SyntaxError, NameError):
                    print(f'Not a number, try again.\n')
                    print(basic_repr_player_actions(action_types, self))

            action_type = action_types[idx]
        else:
            # If there is only one action type, take it by default
            action_type = list(action_types)[0]

        choices = [i[1] for i in allowed_actions if i[0] == action_type]

        if len(choices) == 1 and action_type != 'PlayFromDeck':
            return action_type, choices[0]

        if action_type == 'Attack':
            while True:
                # Get the input from user
                move_input = input(f"Enter the card(s) indexes for {action_type} as numbers separated by space (1-{len(self.hand)}): ")

                # Parse the input
                try:
                    move = move_input.split()
                    selected = []

                    for card in move:
                        suit, value = self.hand[int(card)-1].suit, self.hand[int(card)-1].value
                        selected.append((int(suit), int(value)))

                    # Validate cards
                    if all(card in choices for card in selected):

                        if len(selected) > 1:
                            values = [card[1] for card in selected]
                            if len(set(values)) == 1:
                                return action_type, selected
                            else:
                                print("Error: When playing multiple cards, all must have the same value.\n")
                                continue
                        else:
                            return action_type, selected[0]
                    else:
                        invalid_cards = [card for card in selected if card not in choices]
                        print(f"Invalid card(s): {invalid_cards}")
                        valid_cards = " ".join(f"{c[0]},{c[1]}" for c in choices)
                        print(f"Valid cards are: {valid_cards}")
                except (ValueError, IndexError, SyntaxError, NameError):
                    print("Input error. Please enter the indexes of the cards you want to play as numbers separated by spaces.\n")

        elif action_type == 'Defend':
            while True:
                # Get the input from user
                move_input = input(f"Enter the card pair(s) indexes for {action_type} as tuples separated by space. (played_card,card_on_table): ")

                # Parse the input
                try:
                    move = move_input.split()
                    played_cards = []
                    cards_killed = []

                    for player_input in move:
                        played_card_idx, card_to_kill_idx = player_input.split(',')
                        played_card = self.hand[int(played_card_idx)-1]
                        card_to_kill = game_state.cards_to_defend[int(card_to_kill_idx)-1]

                        played_cards.append(played_card)
                        cards_killed.append(card_to_kill)

                        print("Played cards:", played_cards, "and killed cards:", cards_killed)
                    print(choices[:], "and you chose", played_cards)

                    # Validate cards
                    valid_cards = [choice[1] for choice in choices]
                    if all(card in valid_cards for card in played_cards):
                        return action_type, (played_cards, cards_killed)
                    else:
                        invalid_cards = [card for card in played_cards if card not in valid_cards]
                        print(f"Invalid card(s): {invalid_cards}")
                        valid_cards = " ".join(f"{c[1]} ({c[1].suit, c[1].value}),{c[0]} ({c[0].suit, c[0].value})" for c in choices)
                        print(f"Valid cards are: {valid_cards}")

                #     if all(card in choices for card in played_cards):
                #
                #         if len(played_cards) > 1:
                #             values = [card[1] for card in selected]
                #             if len(set(values)) == 1:
                #                 return action_type, selected
                #             else:
                #                 print("Error: When playing multiple cards, all must have the same value.\n")
                #                 continue
                #         else:
                #             return action_type, selected[0]
                #     else:
                #         invalid_cards = [card for card in selected if card not in choices]
                #         print(f"Invalid card(s): {invalid_cards}")
                #         valid_cards = " ".join(f"{c[0]},{c[1]}" for c in choices)
                #         print(f"Valid cards are: {valid_cards}")
                except (ValueError, IndexError, SyntaxError, NameError):
                    print("Invalid format. Please enter cards as 'suit,value' separated by spaces.")

        elif action_type == 'PlayFromDeck':
            to_defend = []
            deck_card = None
            can_play = None

            print("Choices are", choices, "allowed actions are", allowed_actions)

            for choice in choices:
                # This means that the card is not playable on any card on the table
                if choice is None:
                    continue
                deck_card = choice[1]
                to_defend.append(choice[0])

            # Check if the drawn card can be used to fall a card on the table
            for action in allowed_actions:
                if action[0] == 'PlayFromDeck':
                    if action[3]:
                        can_play = True
                        break
                    else:
                        can_play = False

            while True:
                print("Card drawn from deck is", repr(deck_card))
                assert can_play is not None

                # Check if the drawn card can be used to fall a card on the table
                if not can_play:
                    print("The drawn card can't be used to fall a card on the table.")

                # If there is only one card on the table to fall, try to fall it
                if len(to_defend) == 1:
                    return action_type, (deck_card, to_defend[0]), can_play

                move_input = input(f"Enter the card you want to kill with the drawn card as a tuple [♣♠♥♦] 1 - 4, 2 - 14: ")

                # Parse the input
                try:
                    suit, value = move_input.split(',')

                    for card in to_defend:
                        if card.suit == int(suit) and card.value == int(value):
                            print(action_type, (deck_card, card), can_play)

                            return action_type, (deck_card, card), can_play

                    # Fixed error message formatting
                    valid_cards = " ".join(f"{c[0]}{c[1]}" for c in choices)
                    print(f'Not valid, try again, the choices are [{valid_cards}]')

                except (ValueError, IndexError, SyntaxError, NameError):
                    print("Invalid format. Please enter cards as 'suit,value' separated by spaces.\n")


        elif action_type in ['Take', 'PassAttack']:
            return action_type, None

        elif action_type == 'ThrowCards':
            idx = eval(input(f'Choose throw from {choices}: '))
            return action_type, choices[idx]

        else:
            raise NotImplementedError
        raise NotImplementedError