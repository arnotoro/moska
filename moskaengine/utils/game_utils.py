from moskaengine.game.deck import StandardDeck

from collections import deque
import os
import csv
import uuid
from pathlib import Path
import time

def state_as_vector(game_state, file_format ="numpy"):
    # The current game state as a vector representation. More details: roadmap.md
    state_data = _game_state_as_vector(game_state)

    # The opponent's cards as a vector representation. This is from the perspective of the current player.
    opponent_data = _opponent_cards_as_vector(game_state)

    return state_data, opponent_data

def save_game_vector(state_data, opponent_data, folder = "game_vectors", file_format = "csv"):
    # Save the game to a Numpy file
    if file_format == "numpy":
        _save_game_vector_numpy(state_data, opponent_data, folder)
    elif file_format == "csv":
        _save_game_vector_csv(state_data, opponent_data, folder)
    else:
        # Print the game state
        print(state_data, opponent_data)

def save_game_state_vector_batch(state_data, batch_number, folder_name = "vectors"):
    save_folder = os.path.join(f"../{folder_name}/states")
    os.makedirs(save_folder, exist_ok=True)
    file_name = os.path.join(save_folder, f"states_results_batch_{batch_number}_{uuid.uuid4()}.csv")
    with open(file_name, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(state_data)

def save_opponent_vector_batch(opponent_data, batch_number, folder_name = "vectors"):

    os.makedirs(save_folder, exist_ok=True)
    file_name = os.path.join(save_folder, f"opponent_results_batch_{batch_number}_{uuid.uuid4()}.csv")
    with open(file_name, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(opponent_data)

def _save_game_vector_numpy(state_data, opponent_data, folder_name = "game_data"):
    pass

def _save_game_vector_csv(state_data, opponent_data, folder_name = "game_data"):
    """
    Save the game state and opponent's cards to a CSV file.
    """

    # Create the folders if they don't exist
    root_folder = get_project_root()

    parent_folder = _create_folders(root_folder, folder_name)

    # With a random file name for state data
    file_name = f"state_data_{uuid.uuid4()}.csv"
    file_path = os.path.join(root_folder, parent_folder, "states", file_name)
    # print(state_data)
    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)
        for turn in state_data:
            writer.writerow(turn)

    file_name = f"opponent_data_{uuid.uuid4()}.csv"
    file_path = os.path.join(root_folder, parent_folder, "opponent", file_name)
    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)

        for turn in opponent_data:
            writer.writerow(turn)

def _game_state_as_vector(game_state):
    """
    Convert the game state to a vector representation.
    """
    vector = []
    killed_list = []
    turn_history = []
    actions = {
        "Attack": 1,
        "Defend": 2,
        "PlayFromDeck": 3,
        "ThrowCards": 4,
        "PassAttack": 5,
        "TakeDefend": 6,
        "TakeAll": 7,
    }

    # The cards left in the deck
    vector += [len(game_state.deck.cards)]

    # The cards on the table
    vector += _encode_cards(game_state.cards_to_defend)

    # The discarded cards i.e., not in the game anymore
    for item in game_state.cards_killed:
        killed_list.extend(item)
    vector += _encode_cards(killed_list)

    # The current player idx (1-4)
    vector += [game_state.players.index(game_state.player_to_play) + 1]

    # Current player's hand
    vector += _encode_cards(game_state.player_to_play.hand)

    # Number of cards in each player's hand
    for player in game_state.players:
        vector += [len(player.hand)]

    # Turn number
    vector += [game_state.n_turns]

    # Move history for the last N turns, default 5
    turn_history = deque([[0, 0] + [0] * 52 for _ in range(game_state.N_HISTORY)], maxlen=game_state.N_HISTORY)
    # For the first turn
    cards = []
    for idx, move in enumerate(game_state.history[-game_state.N_HISTORY:]):
        # Action idx (check actions dict)
        action = actions[move[0][0]]
        # Player idx (1-4)
        player = next(idx for idx, pl in enumerate(game_state.players) if pl.name == move[1]) + 1
        # Played card by reference deck
        if isinstance(move[0][1], tuple) or isinstance(move[0][1], list):
            # If the cards in the history are given as a tuple e.g., for Defend or PlayFromDeck
            if move[0][1][1] is None:
                # Special case when PlayFromDeck can't be played on a card on the table
                cards = _encode_cards([move[0][1][0]])
            else:
                cards = _encode_cards([move[0][1][0], move[0][1][1]])
        elif move[0][1] is None:
            pass
        else:
            cards = _encode_cards([move[0][1]])

        turn_history.append([action, player] + cards)

    flat_history = [item for sublist in turn_history for item in sublist]
    vector += list(flat_history)

    return vector

def _opponent_cards_as_vector(game_state):
    """Convert the opponent's cards to a vector representation. This is from the perspective of the current player.
    Used as label data for the model.
    """
    vector = []
    for idx, player in enumerate(game_state.players):
        if player == game_state.player_to_play:
            continue
        # Player idx (1-4)
        vector += [game_state.players.index(player) + 1]
        # Encode the opponent's hand
        vector += _encode_cards(player.hand)

    return vector

def _encode_cards(cards, fill = 0):
    """
    Encode the cards into a vector representation.
    """
    reference_deck = StandardDeck(shuffle = False, perfect_info = True)
    encoded_cards = [fill] * len(reference_deck.cards)
    for card in cards:
        if card.is_unknown:
            # Encode unknown cards as a special value
            # encoded_cards[reference_deck.cards.index(card)] = -1
            pass
        else:
            # Encode the known cards as the index in the non-shuffled deck
            encoded_cards[reference_deck.cards.index(card)] = 1

    return encoded_cards

def _create_folders(root_folder, folder_name):
    """Create the folder `folder_name` in the root of the project and create the subfolders `states` and `opponent"""
    # Format the time as a string (e.g., 20250428_123456)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    parent_folder = os.path.join(root_folder, f"{folder_name}")
    os.makedirs(os.path.join(parent_folder, f"states"), exist_ok=True)
    os.makedirs(os.path.join(parent_folder, "opponent"), exist_ok=True)

    return parent_folder


def get_project_root():
    # This function calculates the absolute path to the root of the project folder
    return Path(__file__).parent.parent.parent
