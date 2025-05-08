from moskaengine.players.abstract_player import AbstractPlayer
from moskaengine.utils.card_utils import basic_repr_game, basic_repr_player_actions


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

        print(f'What does {self} do?')

        action_types = sorted(action_types)

        # Ask the user to choose an action
        while True:
            try:
                # Subtract 1 from the input to get the correct index
                if len(action_types) > 1:
                    idx = int(input(f'Choose action (1 - {len(action_types)}): ')) - 1
                else:
                    idx = int(input(f'Choose action ({len(action_types)}): ')) - 1

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

        choices = [i[1] for i in allowed_actions if i[0] == action_type]

        # Check if the action type is 'ThrowCards' and there is only one choice i.e. no card to throw
        # NOTE: Very unintuitive...
        if len(choices) == 1 and action_type == 'ThrowCards':
            return action_type, choices[0]

        if action_type == 'Attack':
            while True:
                if any(isinstance(choice, tuple) for choice in choices):
                    print(f"Allowed attack combinations are: {choices}")

                # Get the input from the player
                move_input = input(f"Enter the card(s) indexes for {action_type} as numbers separated by space (1-{len(self.hand)}): ")

                # Parse the input
                try:
                    move = move_input.split()
                    selected = []

                    for card in move:
                        selected.append(self.hand[int(card)-1])

                    # Validate cards
                    if all(card in choices for card in selected):
                        if len(selected) > 1:
                            values = [card.value for card in selected]
                            if len(set(values)) == 1:
                                return action_type, tuple(selected)
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

                    # Validate cards
                    valid_cards = [choice[0] for choice in choices]
                    if all(card in valid_cards for card in played_cards):
                        if len(played_cards) > 1:
                            return action_type, played_cards, cards_killed
                        else:
                            return action_type, (played_cards[0], cards_killed[0])
                    else:
                        invalid_cards = [card for card in played_cards if card not in valid_cards]
                        print(f"Invalid card(s): {invalid_cards}")
                        valid_cards = " ".join(f"{c[0]}" for c in choices)
                        print(f"Valid cards to play are: {valid_cards}")
                except (ValueError, IndexError, SyntaxError, NameError):
                    print("Invalid format. Please enter cards as 'suit,value' separated by spaces.")

        elif action_type == 'PlayFromDeck':
            to_defend = []
            deck_card = None
            can_play = None

            for choice in choices:
                # This means that the card is not playable on any card on the table
                if choice is None:
                    continue
                deck_card = choice[0]
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
                # print("Card drawn from deck is", repr(deck_card))
                assert can_play is not None

                # Check if the drawn card can be used to fall a card on the table
                if not can_play:
                    print("The drawn card can't be used to fall a card on the table.")
                    return action_type, (deck_card, None), can_play

                # If there is only one card on the table to fall, try to fall it
                if len(to_defend) == 1:
                    return action_type, (deck_card, to_defend[0]), can_play

                n = 1
                for i in to_defend:
                    if i is None:
                        continue
                    print(f"{n}. {repr(i)}")
                    n += 1

                print()
                move_input = input(f"Enter the index of the card you want to kill with the drawn card: ")

                # Parse the input
                try:
                    move = move_input.split()

                    selected_card = to_defend[int(move[0])-1]

                    for card in to_defend:
                        if card.suit == selected_card.suit and card.value == selected_card.value:
                            return action_type, (deck_card, card), can_play

                    # Fixed error message formatting
                    valid_cards = " ".join(f"{c[0]}{c[1]}" for c in choices)
                    print(f'Not valid, try again, the choices are [{valid_cards}]')

                except (ValueError, IndexError, SyntaxError, NameError):
                    print("Invalid input, please enter the index of the card you want to kill.\n")


        elif action_type in ['TakeAll', 'TakeDefend', 'PassAttack']:
            return action_type, None

        elif action_type == 'ThrowCards':
            while True:
                # Print the choices for the player
                for i, choice in enumerate(choices):
                    print(f"{i+1}. {repr(choice[0])}")

                move_input = input(f"Enter the index of the card you want to throw to table (multiple seperated by space): ")

                # Parse the input
                try:
                    move = move_input.split()
                    selected = []

                    for card in move:
                        selected.append(choices[int(card)-1])

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
                    print(
                        "Input error. Please enter the indexes of the cards you want to play as numbers separated by spaces.\n")
        else:
            raise NotImplementedError