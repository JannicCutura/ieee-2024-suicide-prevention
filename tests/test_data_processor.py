from src.data_preprocessor import count_words, clean_text


def test_count_words():
    assert count_words("this is a sentence with seven words") == 7


def test_clean_text():
    input_txt = "this <br> is: é, à, ö, ñ, etc. Non-Latin alphabets: 漢 (Chinese), こんにちは (Japanese), به متنی(Persian), etc. and Symbols: ©, ®, €, £, µ, ¥, etc. "
    expected_output = 'this is: , , , , etc. Non-Latin alphabets: (Chinese), (Japanese), (Persian), etc. and Symbols: , , , , , , etc.'
    assert clean_text(input_txt) == expected_output
