import random

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
                weights=list(map(lambda x: x[2], poss_actions))
            )
    else:
        # Weights are not used
        return choose_random(poss_actions)
