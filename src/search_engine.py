from typing import List, Optional, Tuple

from invertedindex import InvertedIndex, SearchState, MyWord
from tfidf_model import TfidfModel


class SearchEngine:
    """
    Высокоуровневый фасад для работы с поиском:
    содержит в себе инвертированный индекс и TF-IDF модель.
    """

    def __init__(
        self,
        sentences: List[str],
        calc_word_freq: bool = False,
        tfidf_vectorizer=None,
    ) -> None:
        self.index = InvertedIndex(sentences, calc_word_freq=calc_word_freq)
        self.tfidf = TfidfModel(sentences, vectorizer=tfidf_vectorizer)

    def search(self, search_word: str, state: Optional[SearchState] = None) -> SearchState:
        """
        Поиск с использованием InvertedIndex и состояния SearchState.
        """
        if state is None:
            state = SearchState()
        return self.index.search(search_word, state)

    def get_top_words_with_tfidf(self, n: int) -> List[Tuple[MyWord, float]]:
        """
        Удобный метод: top-N слов по частоте + соответствующие им TF-IDF значения.
        """
        top_words: List[MyWord] = self.index.getTopWordFrequency(n)
        scores = self.tfidf.get_words_tfidf(w.word for w in top_words)
        return list(zip(top_words, scores))

