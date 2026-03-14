"""
Модуль для непересекающейся кластеризации на основе TF-IDF.

В отличие от PMI-based кластеризации (cluster_analyzer.py),
этот модуль назначает каждому предложению ровно одно релевантное слово
на основе TF-IDF метрики.
"""

from typing import Optional, Dict, Set
import numpy as np
from scipy.sparse import csr_matrix


class ExclusiveClusterer:
    """
    Непересекающаяся кластеризация предложений по релевантным словам.

    Каждому предложению назначается ровно одно слово-представитель,
    выбранное на основе TF-IDF × log(freq).

    Пример использования:
        clusterer = ExclusiveClusterer(tfidf_matrix, feature_names, word_freqs)
        clusters = clusterer.cluster(n=10000)
        # {'data': {0, 5, 23}, 'python': {1, 12}, ...}
    """

    def __init__(
        self,
        tfidf_matrix: csr_matrix,
        feature_names: np.ndarray,
        word_freqs: Dict[str, int],
    ):
        """
        Инициализация кластеризатора.

        Args:
            tfidf_matrix: TF-IDF матрица размера (n_docs, n_features).
            feature_names: Массив имён признаков (слов) из векторизатора.
            word_freqs: Словарь {слово: частота} из инвертированного индекса.
        """
        self._tfidf_matrix = tfidf_matrix
        self._feature_names = feature_names
        self._word_freqs = word_freqs
        self._n_docs, self._n_features = tfidf_matrix.shape

        # Предварительно вычисляем веса для всех слов
        self._word_weights = self._compute_word_weights()

    def _compute_word_weights(self) -> np.ndarray:
        """
        Вычислить веса для всех слов на основе их частоты.

        Returns:
            Массив весов размера (n_features,).
        """
        weights = np.zeros(self._n_features, dtype=np.float64)
        for i, word in enumerate(self._feature_names):
            freq = self._word_freqs.get(word, 0)
            weights[i] = np.log(freq + 1)
        return weights

    def cluster(
        self,
        n: Optional[int] = None,
        excluded_words: Optional[Set[str]] = None,
    ) -> Dict[str, Set[int]]:
        """
        Выполнить непересекающуюся кластеризацию.

        Использует метрику TF-IDF × log(freq) для релевантности.

        Args:
            n: Количество предложений для обработки. Если None — все.
            excluded_words: Множество слов для исключения из кандидатов.
                           Если для предложения лучшее слово исключено, берётся следующее.

        Returns:
            Словарь {слово: множество индексов предложений}.
            Каждое предложение принадлежит только одному кластеру.
        """
        n_docs = n if n is not None else self._n_docs

        if n_docs == 0 or self._n_features == 0:
            return {}

        # Берём подмножество матрицы
        matrix = self._tfidf_matrix[:n_docs]

        # Вычисляем скоры: TF-IDF × log(freq)
        scores = matrix.multiply(self._word_weights).tocsr()

        # Получаем максимальные скоры для проверки
        max_scores = np.asarray(scores.max(axis=1).toarray()).flatten()

        # Собираем результат: слово -> множество индексов предложений
        from collections import defaultdict
        clusters = defaultdict(set)

        # Индексы исключённых слов для быстрого доступа
        excluded_indices = set()
        if excluded_words:
            for i, word in enumerate(self._feature_names):
                if word.lower() in {w.lower() for w in excluded_words}:
                    excluded_indices.add(i)

        for doc_idx in range(n_docs):
            # Проверяем, что в предложении есть хоть одно слово
            if max_scores[doc_idx] == 0:
                continue

            # Получаем все слова и их скоры для этого предложения
            row = scores.getrow(doc_idx)
            word_indices = row.indices
            word_scores = row.data

            if len(word_indices) == 0:
                continue

            # Сортируем по убыванию скора
            sorted_order = np.argsort(word_scores)[::-1]

            # Находим первое слово, которое не исключено
            assigned = False
            for idx in sorted_order:
                word_idx = word_indices[idx]
                if word_idx not in excluded_indices:
                    word = self._feature_names[word_idx]
                    clusters[word].add(doc_idx)
                    assigned = True
                    break

            # Если все слова исключены - пропускаем предложение

        return dict(clusters)

    def get_top_clusters(
        self,
        n: int = 20,
        min_cluster_size: int = 1,
    ) -> Dict[str, Set[int]]:
        """
        Получить топ-N крупнейших кластеров.

        Args:
            n: Количество кластеров для возврата.
            min_cluster_size: Минимальный размер кластера для фильтрации.

        Returns:
            Словарь {слово: множество индексов} для топ-N кластеров.
        """
        # Сначала кластеризуем всё
        clusters = self.cluster()

        # Фильтруем по минимальному размеру
        filtered = {
            word: indices
            for word, indices in clusters.items()
            if len(indices) >= min_cluster_size
        }

        # Сортируем по размеру и берём топ-N
        sorted_clusters = sorted(
            filtered.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )[:n]

        return dict(sorted_clusters)
