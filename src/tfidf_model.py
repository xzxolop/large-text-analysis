from typing import Iterable, List, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfModel:
    """
    Отвечает только за хранение и работу с TF-IDF представлением корпуса.
    Не знает об инвертированном индексе и структурах вроде MyWord.
    """

    def __init__(
        self,
        sentences: List[str],
        vectorizer: Optional[TfidfVectorizer] = None,
    ) -> None:
        self._vectorizer = vectorizer or TfidfVectorizer(
            stop_words="english",
            min_df=2,
        )
        self._matrix = self._vectorizer.fit_transform(sentences)

    @property
    def vectorizer(self) -> TfidfVectorizer:
        return self._vectorizer

    def get_mean_tfidf(self) -> List[Tuple[str, float]]:
        """
        Средний TF-IDF по каждому слову (как в старом getMeanTfidf).
        """
        matrix = self._matrix
        feature_names = self._vectorizer.get_feature_names_out()
        sum_tfidf = np.array(matrix.sum(axis=0)).flatten()

        _, cols = matrix.nonzero()
        word_frequencies = np.bincount(cols, minlength=len(feature_names))
        word_frequencies_safe = np.where(word_frequencies == 0, 1, word_frequencies)

        normalized_scores = sum_tfidf / word_frequencies_safe
        result = list(zip(feature_names, normalized_scores))
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def get_word_tfidf(self, word: str) -> float:
        """
        Средний TF-IDF по документам для одного слова.
        """
        feature_names = self._vectorizer.get_feature_names_out()

        try:
            word_index = np.where(feature_names == word)[0][0]
        except IndexError:
            return 0.0

        word_tfidf = self._matrix[:, word_index].toarray().flatten()
        nonzero_values = word_tfidf[word_tfidf > 0]

        if len(nonzero_values) == 0:
            return 0.0

        return float(np.mean(nonzero_values))

    def get_words_tfidf(self, words: Iterable[str]) -> List[float]:
        """
        TF-IDF для набора слов (список средних значений).
        """
        return [self.get_word_tfidf(w) for w in words]

