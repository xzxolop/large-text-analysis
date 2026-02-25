from typing import Iterable, List, Optional, Tuple

from core.inverted_index import InvertedIndex, SearchState, MyWord
from core.tfidf_model import TfidfModel


class SearchEngine:
    """
    Высокоуровневый фасад для работы с поиском:
    скрывает внутри себя инвертированный индекс и TF-IDF модель.
    """

    def __init__(
        self,
        sentences: List[str],
        calc_word_freq: bool = False,
        tfidf_vectorizer=None,
    ) -> None:
        self._index = InvertedIndex(sentences, calc_word_freq=calc_word_freq)
        self._tfidf = TfidfModel(sentences, vectorizer=tfidf_vectorizer)

# Функции из класса InvertedIndex

    def search(self, search_word: str, state: Optional[SearchState] = None) -> SearchState:
        """Поиск с использованием InvertedIndex и состояния SearchState."""
        if state is None:
            state = SearchState()
        return self._index.search(search_word, state)

    def get_top_word_frequency(self, n: int | None = None):
        """Top-N самых частых слов по индексу."""
        return self._index.get_top_word_frequency(n)

    def get_least_frequent_words(self, n: int | None = None):
        """Top-N наименее частых слов по индексу."""
        return self._index.get_least_frequent_words(n)

    def get_words_by_frequency(self, freq: int) -> list[MyWord]:
        """Вернуть все слова с указанной частотой."""
        return self._index.get_words_by_frequency(freq)

    def print_top_word_frequency(self, n: int | None = None) -> None:
        """Печатает top-N самых частых слов по индексу."""
        self._index.print_top_word_frequency(n)

# Функции из класса TfidfModel

    def get_top_words_with_tfidf(self, n: int) -> List[Tuple[MyWord, float]]:
        """Top-N слов по частоте + соответствующие им TF-IDF значения."""
        top_words: List[MyWord] = self._index.get_top_word_frequency(n)
        scores = self._tfidf.get_words_tfidf(w.word for w in top_words)
        return list(zip(top_words, scores))

    def get_word_tfidf(self, word: str) -> float:
        """TF-IDF для одного слова."""
        return self._tfidf.get_word_tfidf(word)

    def get_words_tfidf(self, words: Iterable[str]) -> List[float]:
        """TF-IDF для набора слов."""
        return self._tfidf.get_words_tfidf(words)
