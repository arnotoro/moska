import csv
from moskaengine.utils.game_utils import StandardDeck

# Standard 52-card deck
deck = StandardDeck(shuffle=False, perfect_info=True)

# Path to CSV
csv_path = "./vectors/opponent/opponents.csv"

# Open and read the CSV line by line
with open(csv_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for line_num, row in enumerate(reader, 1):
        # Convert string values to ints
        vector = list(map(int, row))

        # Find cards corresponding to 1s
        hand = [card for bit, card in zip(vector, deck) if bit == 1]

        print(f"Line {line_num} hand: {hand}")