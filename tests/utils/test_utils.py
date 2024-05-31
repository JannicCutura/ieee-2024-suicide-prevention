from src.utils.util_functions import key_with_max_value, ordered_values


def test_key_with_max_value_standard():
    my_dict = {'a': 13, 'b': 12, 'c': 0}
    assert key_with_max_value(my_dict) == 'a'


def test_key_with_max_value_empty():
    my_dict = dict()
    assert key_with_max_value(my_dict) is None


def test_ordered_values():
    my_dict = {'ideation': 13, 'indicator': 12, 'behavior': 5, 'attempt':6}
    assert ordered_values(my_dict) == [12, 13, 5, 6]
