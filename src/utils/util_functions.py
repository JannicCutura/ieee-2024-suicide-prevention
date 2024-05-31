def key_with_max_value(d):
    """
    Returns the key with the largest value in the dictionary.

    Parameters:
    d (dict): The dictionary to search.

    Returns:
    key: The key with the largest value.
    """
    if not d:
        return None  # Return None if the dictionary is empty

    # Use max() with key argument to find the key with the maximum value
    return max(d, key=lambda k: d[k])


def ordered_values(d):
    """
    Returns a list of values from the dictionary ordered by the specified keys.

    Parameters:
    d (dict): The dictionary containing the keys 'indicator', 'ideation', 'behavior', and 'attempt'.

    Returns:
    list: A list of values ordered by the keys ['indicator', 'ideation', 'behavior', 'attempt'].
    """
    # Define the required order of keys
    required_order = ['indicator', 'ideation', 'behavior', 'attempt']

    # Extract values in the required order
    ordered_values_list = [round(d[key],5) for key in required_order]

    return ordered_values_list