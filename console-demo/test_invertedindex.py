from invertedindex import InvertedIndex
from invertedindex import WordFrequency

sentances = [
    "my name is ononim",
    "ononim is very rare name",
    "i play in computer",
    "computer is complex system",
    "computer is not ononim"
]

index = InvertedIndex(sentances, calc_word_freq=True)
sf_global = index.get_searched_frequency()
#for x in sf_global:
#    print(x)

def test_invariant_searched_frequency1():
    """
    Проверка не ломается ли инвариант searched_frequency
    """
    sf1 = index.get_searched_frequency()
    index.searchWith("computer")
    index.clearState()
    sf2 = index.get_searched_frequency()

    assert sf1 == sf2

def test_searchWith_word_exists():
    """
    Тест возвращаемого значения функции searchWith: Поиск существующего слова
    """
    res = index.searchWith("computer")
    expected = {2,3,4}
    
    index.clearState()
    assert res == expected

def test_searchWith_word_nonexists():
    """
    Тест возвращаемого значения функции searchWith: Поиск несуществующего слова
    """
    res = index.searchWith("non_exist_word")
    expected = {}
    
    index.clearState()
    assert res == expected

def test_word_frequency():
    index.get_searched_frequency()


def test_calculate_frequency():
    expected = []
    # TODO: можно сделать <число встреч, список слов> чтобы не зависить от порядка
    expected.append(WordFrequency("is", 4))
    expected.append(WordFrequency("ononim", 3))
    expected.append(WordFrequency("computer", 3))
    expected.append(WordFrequency("name", 2))
    expected.append(WordFrequency("my", 1))
    expected.append(WordFrequency("very", 1))
    expected.append(WordFrequency("rare", 1))
    expected.append(WordFrequency("i", 1))
    expected.append(WordFrequency("play", 1))
    expected.append(WordFrequency("in", 1))
    expected.append(WordFrequency("complex", 1))
    expected.append(WordFrequency("system", 1))
    expected.append(WordFrequency("not", 1))

    sf = index.get_searched_frequency()
    assert len(sf) == len(expected)
    assert sf == expected
    index.clearState()

def test_calculate_frequency_word_nonexists():
    expected = []
    sf = index.searchWith("non_exist_word")
    sf = index.get_searched_frequency()
    assert sf == expected
    index.clearState()

def test_calculate_frequency_word_exists():
    expected = []
    
    expected.append(WordFrequency("i", 1))
    expected.append(WordFrequency("play", 1))
    expected.append(WordFrequency("in", 1))
    expected.append(WordFrequency("computer", 1))

    index.searchWith("play")
    sf = index.get_searched_frequency()

    assert sf == expected
    index.clearState()