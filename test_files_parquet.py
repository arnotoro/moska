import os
import re
import pandas as pd

directory = "./vectors_10k/opponent"

pattern = re.compile(r"opponents_batch_([a-f0-9\-]{36})_(\d+)\.parquet")

for filename in os.listdir(directory):
    match = pattern.match(filename)
    if match:
        uuid_part = match.group(1)
        number_part = match.group(2)
        print(f"Found a match: {filename} with UUID {uuid_part} and number {number_part}")

        filepath = os.path.join(directory, filename)

        # Check file size before reading
        if os.path.getsize(filepath) == 0:
            print(f"{filename} is empty on disk.")
            continue

        # Read the parquet file
        df = pd.read_parquet(filepath)

        # Check if DataFrame is empty
        if df.empty:
            print(f"{filename} is an empty DataFrame.")
            continue

        print(df)

        # Check for non-zero values
        if (df != 0).any().any():
            print(f"{filename} contains non-zero values.")
        else:
            print(f"{filename} contains only zeros.")
