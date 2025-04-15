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
    if len(poss_actions[0]) == 3:
        # Weights are used
        return choose_random(
                list(map(lambda x: (x[0], x[1]), poss_actions)),
                weights = list(map(lambda x: x[2], poss_actions))
            )
    else:
        # Weights are not used
        return choose_random(poss_actions)


def basic_repr_game(game_state):
    """Returns a basic representation of the game state"""
    # print("Hello from card_utils")
    string = f"Trump card: {game_state.trump_card}\n"
    string += f"Deck left: {len(game_state.deck)}\n"
    for pl in game_state.players:
        string += f"{pl.name}{' (TG)' if pl is game_state.defender else ''}"
        string += " " * max(16 - len(string.split("\n")[-1]), 1)
        string += f" : {pl.hand}\n"

    string += f"Cards to defend : {game_state.cards_to_defend}\n"
    string += f"Killed cards : {game_state.pairs_finished}\n"

    return string

def basic_repr_player_actions(action_types, player):
    """Returns a basic representation of the player actions"""
    string = f"Allowed actions for {player}:\n"
    n = 1
    # Sort actions
    for i in sorted(action_types):
        if i in ['Attack', 'Defend', 'Reflect', 'ReflectTrump']:
            string += f"{n}. {str(i)}\n"
        else:
            string += f"{n}. {str(i)}\n"
        n += 1

    return string