import time
import os
import csv
import uuid
from collections import deque
from pathlib import Path

# Moskaengine improts
from ..game.deck import StandardDeck


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

    # The cards left in the deck
    vector.append(len(game_state.deck.cards))

    # The trump suit (multi-hot encoding of 4 suits) (clubs, spades, hearts, diamonds)
    # 0 = clubs, 1 = spades, 2 = hearts, 3 = diamonds
    # The trump suit is encoded as a one-hot vector of length 4
    # e.g., [1, 0, 0, 0] for clubs, [0, 1, 0, 0] for spades, etc.
    trump_suit = game_state.trump_card.suit
    trump_vector = [0] * 4
    if trump_suit == 1:
        trump_vector[0] = 1  # Clubs
    elif trump_suit == 2:
        trump_vector[1] = 1  # Spades
    elif trump_suit == 3:
        trump_vector[2] = 1  # Hearts
    elif trump_suit == 4:
        trump_vector[3] = 1  # Diamonds
    vector.extend(trump_vector)

    # The cards on the table
    vector.extend(_encode_cards(game_state.cards_to_defend))

    # The discarded cards i.e., not in the game anymore
    vector.extend(_encode_cards(game_state.cards_discarded))

    # The current player idx, multi-hot encoding of 4 players
    # 1 = player 1, 2 = player 2, 3 = player 3, 4 = player 4
    player_id_vector = [0] * len(game_state.players)
    player_id_vector[game_state.players.index(game_state.player_to_play)] = 1
    vector.extend(player_id_vector)

    # Current player's hand
    vector.extend(_encode_cards(game_state.player_to_play.hand))

    # Number of cards in each player's hand
    vector.extend(len(player.hand) for player in game_state.players)

    # Turn number
    vector.append(game_state.n_turns)

    # Move history for the last N turns, default 5
    actions = {
        "Attack": 1,
        "Defend": 2,
        "PlayFromDeck": 3,
        "ThrowCards": 4,
        "Skip": 5,
        "TakeDefend": 6,
        "TakeAll": 7,
    }
    n_actions = len(actions)
    n_players = len(game_state.players)
    history_len = (n_actions + n_players+ 52)  # action_type one-hot, player_idx one-hot + 52 cards multi-hot

    
    turn_history = deque([[0] * history_len for _ in range(game_state.N_HISTORY)], maxlen=game_state.N_HISTORY)

    # Map player names to indices (0-3)
    player_indices = {pl.name: idx for idx, pl in enumerate(game_state.players)}

    for move in game_state.history[-game_state.N_HISTORY:]:
        move_type, move_player = move[0][0], move[1]
        move_cards = move[0][1]

        # Encode action as one-hot vector
        action_idx = actions.get(move_type, -1)
        action_vec = [0] * n_actions
        if 0 <= action_idx < n_actions:
            action_vec[action_idx - 1] = 1

        # Encode player index as one-hot vector
        player_idx = player_indices.get(move_player, -1)
        player_vec = [0] * n_players
        if 0 <= player_idx < n_players:
            player_vec[player_idx] = 1

        # Cards played in the move as a multi-hot vector
        if move_cards is None:
            cards_vec = [0] * 52
        elif isinstance(move_cards, (tuple, list)):
            if not move_cards[1]:
                cards_vec = _encode_cards([move_cards[0]])
            else:
                cards_vec = _encode_cards([move_cards[0], move_cards[1]])
        else:
            cards_vec = _encode_cards([move_cards])

        # Combine all vectors
        combined_vec = action_vec + player_vec + cards_vec
        turn_history.append(combined_vec)

    # Flatten the history
    flat_history = [item for turn in turn_history for item in turn]
    assert len(flat_history) == game_state.N_HISTORY * history_len, f"The length of the history is not correct: {len(flat_history)}"

    vector.extend(flat_history)

    return vector

def _opponent_cards_as_vector(game_state):
    """Convert the opponent's cards to a vector representation. This is from the perspective of the current player.
    Used as label data for the model.
    """
    vector = []
    player_to_play = game_state.player_to_play
    players = game_state.players

    opponents = [pl for pl in players if pl != player_to_play]
    opponents = sorted(opponents, key=lambda p: game_state.player_ids[p])

    for opponent in opponents:
        # Player idx (1-4)
        # vector.append(game_state.player_ids[opponent])
        # Encode the opponent's hand
        vector.extend(_encode_cards(opponent.hand))

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

def check_unique_game_state(game_state):
    """Check that all cards in the game state (players' hands + deck) are unique."""
    all_cards = []

    # Collect cards from all players' hands
    for player in game_state.players:
        all_cards.extend(player.hand)

    # Collect cards from the deck
    all_cards.extend(game_state.deck.cards)

    # Build a set of (suit, value) pairs
    card_tuples = [(card.suit, card.value) for card in all_cards]

    # Check for duplicates
    if len(card_tuples) != len(set(card_tuples)):
        # Find which cards are duplicated (optional)
        from collections import Counter
        counts = Counter(card_tuples)
        duplicates = [card for card, count in counts.items() if count > 1]
        raise ValueError(f"Duplicate cards found: {duplicates}")