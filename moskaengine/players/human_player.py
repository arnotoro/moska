from moskaengine.players.abstract_player import AbstractPlayer


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

        # Print the allowed actions
        print()
        print([c for c in self.hand])
        print(f'Allowed actions for {self} are:')
        print([i[0] for i in allowed_actions])
        print([(i[0], i[1]) for i in allowed_actions])
        print(f'What does {self} do?')

        if len(action_types) > 1:
            action_types = sorted(action_types)
            idx = eval(input(f'Choose action from {action_types}: '))
            action_type = action_types[idx]
        else:
            action_type = list(action_types)[0]

        choices = [i[1] for i in allowed_actions if i[0] == action_type]
        if len(choices) == 1:
            return (action_type, choices[0])

        if action_type in ['Attack', 'Defend', 'Reflect', 'ReflectTrump']:
            while True:
                suit = int(input(f'Suit of the {action_type} card [♣♠♥♦]: '))
                value = int(input(f'Value of the {action_type} card [23456789*JQKA]: '))
                print("Choices are", choices, "and you chose", (suit, value))
                if (suit, value) in choices:
                    break
                # Fixed error message formatting
                valid_cards = " ".join(f"{c[0]}{c[1]}" for c in choices)
                print(f'Not valid, try again, the choices are [{valid_cards}]')
            return (action_type, (suit, value))
        elif action_type in ['Take', 'PassAttack']:
            return (action_type, None)
        elif action_type == 'ThrowCards':
            idx = eval(input(f'Choose throw from {choices}: '))
            return (action_type, choices[idx])
        else:
            raise NotImplementedError
        raise NotImplementedError