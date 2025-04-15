def has_combinations(choices):
    """Check if there are combinations of cards available to play and return them"""
    # Check if the list is empty
    if not choices:
        return None

    # Check if the list contains combination tuples and return them
    return [item for item in choices if isinstance(item, tuple) and all(isinstance(subitem, tuple) for subitem in item)]