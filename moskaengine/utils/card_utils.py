import random

from moskaengine.game.deck import suit_to_symbol


def choose_random(lst, weights=None):
    """Choose a random element"""
    if weights is None:
        return random.choice(list(lst))
    else:
        return random.choices(list(lst))[0]

def choose_random_action(poss_actions):
    """Returns a random action from poss_actions with and without weights"""
    # Check if we used weights
    if len(poss_actions[0]) >= 3:
        # Get weights
        weights = list(map(lambda x: x[2], poss_actions))
        actions = []
        for action in poss_actions:
            if len(action) == 4:
                actions.append((action[0], action[1], action[3]))
            else:
                actions.append((action[0], action[1]))
        chosen_action = choose_random(actions, weights)
        return chosen_action
    else:
        # NOTE: Might not work right now
        # Weights are not used
        return choose_random(poss_actions)

def basic_repr_game(game_state):
    """Returns a basic representation of the game state"""
    # print("Hello from card_utils")
    string = f"\nPlayer to play: {game_state.player_to_play.name}\n"
    string += f"Trump card: {game_state.trump_card}\n"
    string += f"Deck left: {len(game_state.deck)}\n"
    for pl in game_state.players:
        string += f"{pl.name}{' (TG)' if pl is game_state.defender else ''}"
        string += " " * max(16 - len(string.split("\n")[-1]), 1)
        if game_state.perfect_info:
            string += f" : {pl.hand}\n"
        else:
            if pl == game_state.player_to_play:
                string += f" : {pl.hand}\n"
            else:
                # TODO: Change the logic to be handled elsewhere
                # Filter and display public and non-public cards
                formatted_hand = []
                for card in pl.hand:
                    if card.is_public:
                        formatted_hand.append(str(card))  # Use the card's string representation
                    else:
                        formatted_hand.append("-X")  # Placeholder for non-public cards

                hand_str = "[" + ", ".join(formatted_hand) + "]" if formatted_hand else "[]"
                string += f" : {hand_str}\n"

    string += f"Cards to defend : {game_state.cards_to_defend}\n"
    string += f"Killed cards : {game_state.cards_killed}\n"

    return string

def basic_repr_player_actions(action_types, player):
    """Returns a basic representation of the current player actions. Used in player class."""
    string = f"Allowed actions for {player}:\n"
    n = 1

    for i in action_types:
        string += f"{n}. {str(i)}\n"
        n += 1

    return string

def game_action_repr(player_to_play, action):
    """Returns a basic representation of the played action. Used in engine."""
    action_type = action[0]

    if action_type == "Attack":
        # Check if the action is a list of tuples (multiple cards) or a single tuple
        if len(action[1]) > 1:
            return f"Player {player_to_play.name} chose action: '{action_type}' with cards {', '.join(str(card) for card in action[1])}"
        else:
            return f"Player {player_to_play.name} chose action: '{action_type}' with card {action[1]}"
    elif action_type == "Defend":
        if len(action[1]) > 2:
            return f"Player {player_to_play.name} chose action: '{action_type}' to defend cards {action[1][0]} with cards {', '.join(str(card) for card in action[1][1:])}"
        else:
            return f"Player {player_to_play.name} chose action: '{action_type}' to defend card {action[1][1]} with card {action[1][0]}"
    elif action_type == "ThrowCards":
            return f"Player {player_to_play.name} chose action: '{action_type}' with cards {action[1]}"
    elif action_type == "PlayFromDeck":
        return f"Player {player_to_play.name} chose action: '{action_type}' and the drawn card was {action[1][0]}"
    elif action_type == "TakeDefend":
        return f"Player {player_to_play.name} chose action: '{action_type}' to lift all non-defended cards from the table"
    elif action_type == "TakeAll":
        return f"Player {player_to_play.name} chose action: '{action_type}' to lift all the cards from the table"
    else:
        return f"Player {player_to_play.name} chose action: '{action_type}'"