import numpy as np
import openai


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
    ordered_values_list = [round(d[key], 5) for key in required_order]

    return ordered_values_list


def unpack(response: openai.Completion) -> list[dict[str, float]]:
    log_probs = response.choices[0].logprobs.top_logprobs[0]
    prob = {k: np.exp(log_p) / np.sum(np.exp(list(log_probs.values()))) for k, log_p in log_probs.items()}
    out = {'ideation': 0, 'behavior': 0, 'indicator': 0, 'attempt': 0}
    for token, p in prob.items():
        if token.startswith(" be"):
            out['behavior'] = out['behavior'] + prob[token]
        elif token.startswith((" id", "atio")):
            out['ideation'] = out['ideation'] + prob[token]
        elif token.startswith(" at"):
            out['attempt'] = out['attempt'] + prob[token]
        elif token.startswith(" in"):
            out['indicator'] = out['indicator'] + prob[token]
    return out


def unpack_short(response: openai.Completion) -> list[dict[str, float]]:
    log_probs = response.choices[0].logprobs.top_logprobs[0]
    prob = {k: np.exp(log_p) / np.sum(np.exp(list(log_probs.values()))) for k, log_p in log_probs.items()}
    out = {'ideation': 0, 'behavior': 0, 'indicator': 0, 'attempt': 0}
    for token, p in prob.items():
        if token.startswith(" be"):
            out['behavior'] = out['behavior'] + prob[token]
        elif token.startswith((" id", 'i')):
            out['ideation'] = out['ideation'] + prob[token]
        elif token.startswith(" at"):
            out['attempt'] = out['attempt'] + prob[token]
        elif token.startswith(" in"):
            out['indicator'] = out['indicator'] + prob[token]
    return out
