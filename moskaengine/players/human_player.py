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
        print(action_types)
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

        if len(choices) == 1:
            return action_type, choices[0]

        if action_type in ['Attack', 'Defend', 'Reflect', 'ReflectTrump']:
            while True:
                suit = int(input(f'Suit of the {action_type} card [♣♠♥♦] (1 - 4): '))
                value = int(input(f'Value of the {action_type} card [23456789*JQKA] (2 - 14): '))
                # print("Choices are", choices, "and you chose", (suit, value))

                if (suit, value) in choices:
                    break

                # Fixed error message formatting
                valid_cards = " ".join(f"{c[0]}{c[1]}" for c in choices)
                print(f'Not valid, try again, the choices are [{valid_cards}]')

            return action_type, (suit, value)

        elif action_type in ['Take', 'PassAttack']:
            return action_type, None

        elif action_type == 'ThrowCards':
            idx = eval(input(f'Choose throw from {choices}: '))
            return action_type, choices[idx]

        else:
            raise NotImplementedError
        raise NotImplementedError