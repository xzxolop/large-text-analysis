from invertedindex import InvertedIndex
from nltk import sent_tokenize

text = "The English Wikipedia is the primary[a] English-language edition of Wikipedia, an online encyclopedia. It was created by Jimmy Wales and Larry" \
" Sanger on January 15, 2001, as Wikipedia's first edition. English Wikipedia is hosted alongside other language editions by the Wikimedia Foundation, " \
"an American nonprofit organization. Its content, written independently of other editions by volunteer editors known as Wikipedians,[1] is in various" \
" varieties of English while aiming to stay consistent within articles. Its internal newspaper is The Signpost. English Wikipedia is the most read version" \
" of Wikipedia,[2][3] accounting for 48% of Wikipedia's cumulative traffic, with the remaining percentage split among the other languages.[4] The English" \
" Wikipedia has the most articles of any edition, at 7,119,312 as of January 2026.[b] It contains 10.7% of articles in all Wikipedias,[b] although it lacks" \
" millions of articles found in other editions.[1] The edition's one-billionth edit was made on 13 January 2021 by editor Steven Pruitt.[5] English Wikipedia," \
" often as a stand-in for Wikipedia overall, has been praised for its enablement of the democratization of knowledge, extent of coverage, unique structure, " \
"culture, and reduced degree of commercial bias. It has been criticized for exhibiting systemic bias, particularly gender bias against women and ideological " \
"bias.[6][7] While its reliability was frequently criticized in the 2000s, it has improved over time, receiving greater praise in the late 2010s and throughout" \
" the 2020s,[8][6][9][c] having become an important fact-checking site.[10][11] English Wikipedia has been characterized as having less cultural bias than other" \
" language editions due to its broader editor base.[2]"

sentances = sent_tokenize(text)

index = InvertedIndex(sentances, calc_word_freq=True)

def test_invariant_searched_frequency():
    """
    Проверка не ломается ли инвариант searched_frequency
    """
    sf1 = index.get_searched_frequency()
    index.searchWith("English")
    index.clearState()
    sf2 = index.get_searched_frequency()

    assert sf1 == sf2

def test_single_word_search1():
    """
    Поиск одного существующего слова
    """
    index.searchWith("English")
    sf1 = index.get_searched_frequency()
    index.clearState()

    index.searchWith("English")
    sf2 = index.get_searched_frequency()
    index.clearState()

    assert sf1 == sf2

def test_single_word_search2():
    """
    Поиск одного несуществующего слова
    """
    index.searchWith("BAN_ban_BAN")
    sf1 = index.get_searched_frequency()
    index.clearState()

    index.searchWith("BAN_ban_BAN")
    sf2 = index.get_searched_frequency()
    index.clearState()

    assert sf1 == sf2

def test_invariant_searched_frequency2():
    """
    Проверка не ломается ли инвариант searched_frequency
    """
    sf1 = index.get_searched_frequency()
    index.searchWith("English")
    index.clearState()
    sf2 = index.get_searched_frequency()

    assert sf1 == sf2

test_invariant_searched_frequency()
test_single_word_search1()
test_single_word_search2()
test_invariant_searched_frequency2()
