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
    ) -> Dict[str, Set[int]]:
        """
        Выполнить непересекающуюся кластеризацию.

        Использует метрику TF-IDF × log(freq) для релевантности.

        Args:
            n: Количество предложений для обработки. Если None — все.

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

        # Находим слово с максимальным скором для каждого предложения
        best_word_indices = np.asarray(scores.argmax(axis=1)).flatten()

        # Получаем максимальные скоры для проверки
        max_scores = np.asarray(scores.max(axis=1).toarray()).flatten()

        # Собираем результат: слово -> множество индексов предложений
        from collections import defaultdict
        clusters = defaultdict(set)

        for doc_idx, word_idx in enumerate(best_word_indices):
            # Проверяем, что в предложении есть хоть одно слово
            if max_scores[doc_idx] > 0:
                word = self._feature_names[word_idx]
                clusters[word].add(doc_idx)

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
