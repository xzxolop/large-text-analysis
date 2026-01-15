from invertedindex import InvertedIndex

sentances = [
    "my name is ononim",
    "ononim is very rare name",
    "i play in computer",
    "computer is complex system",
    "computer is not ononim"
]

index = InvertedIndex(sentances, calc_word_freq=True)
sf_global = index.get_searched_frequency()
for x in sf_global:
    print(x)

def test_invariant_searched_frequency1():
    """
    Проверка не ломается ли инвариант searched_frequency
    """
    sf1 = index.get_searched_frequency()
    index.searchWith("computer")
    index.clearState()
    sf2 = index.get_searched_frequency()

    assert sf1 == sf2

def test_searchWith_result1():
    """
    Поиск существующего слова
    """
    res = index.searchWith("computer")
    expected = {2,3,4}
    
    index.clearState()
    assert res == expected

def test_searchWith_result2():
    """
    Поиск несуществующего слова
    """
    res = index.searchWith("non_exist_word")
    expected = {}
    
    index.clearState()
    assert res == expected


