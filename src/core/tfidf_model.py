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
            #stop_words="english",
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
        Возвращает TF-IDF для набора слов (список средних значений).
        Оптимизированная версия с векторизованным вычислением.
        """
        feature_names = self._vectorizer.get_feature_names_out()
        feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}

        scores = []
        for word in words:
            if word in feature_to_idx:
                word_index = feature_to_idx[word]
                word_tfidf = self._matrix[:, word_index].toarray().flatten()
                nonzero_values = word_tfidf[word_tfidf > 0]
                if len(nonzero_values) > 0:
                    scores.append(float(np.mean(nonzero_values)))
                else:
                    scores.append(0.0)
            else:
                # Слово не в словаре (например, min_df=2 и слово встречается 1 раз)
                scores.append(0.0)

        return scores

    def get_word_tfidf_in_sentence(self, word: str, sentence_index: int) -> float:
        """
        Возвращает TF-IDF конкретного слова в конкретном предложении.

        Args:
            word: Слово для поиска.
            sentence_index: Индекс предложения (0-based).

        Returns:
            TF-IDF вес слова в указанном предложении, или 0.0 если слово не найдено.
        """
        feature_names = self._vectorizer.get_feature_names_out()

        try:
            word_index = np.where(feature_names == word)[0][0]
        except IndexError:
            return 0.0

        if sentence_index < 0 or sentence_index >= self._matrix.shape[0]:
            return 0.0

        return float(self._matrix[sentence_index, word_index])
