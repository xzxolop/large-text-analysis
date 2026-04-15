"""
Модуль для непересекающейся кластеризации на основе TF-IDF.

В отличие от PMI-based кластеризации (cluster_analyzer.py),
этот модуль назначает каждому предложению ровно одно релевантное слово
на основе TF-IDF метрики.
"""

from typing import Optional, Dict, Set, Iterable, List
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
        # Быстрые отображения для исключения слов
        self._feature_to_idx: Dict[str, int] = {
            str(w): i for i, w in enumerate(self._feature_names)
        }
        self._feature_lower_to_idx: Dict[str, int] = {
            str(w).lower(): i for i, w in enumerate(self._feature_names)
        }

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

    def _excluded_indices(self, excluded_words: Optional[Set[str]]) -> Set[int]:
        """
        Преобразовать excluded_words в индексы фич.

        Делается через словарь feature_lower_to_idx, чтобы не проходить
        по всем feature_names каждый вызов.
        """
        if not excluded_words:
            return set()
        idxs: Set[int] = set()
        for w in excluded_words:
            i = self._feature_lower_to_idx.get(str(w).lower())
            if i is not None:
                idxs.add(i)
        return idxs

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

        return self.cluster_on_indices(
            doc_indices=list(range(n_docs)),
            excluded_words=excluded_words,
        )

    def cluster_on_indices(
        self,
        doc_indices: Iterable[int],
        excluded_words: Optional[Set[str]] = None,
    ) -> Dict[str, Set[int]]:
        """
        Быстрая кластеризация на подмножестве документов без пересборки TF-IDF.

        Алгоритм: для каждого выбранного документа берём ненулевые элементы
        строки CSR и находим максимум TF-IDF × log(freq). Если задан
        excluded_words — выбираем лучший не-исключённый.

        Args:
            doc_indices: индексы документов (предложений) в исходной матрице.
            excluded_words: множество слов, которые нельзя назначать кластером.

        Returns:
            {слово: множество индексов документов} (индексы — исходные).
        """
        from collections import defaultdict

        excluded_indices = self._excluded_indices(excluded_words)

        matrix = self._tfidf_matrix.tocsr()
        indptr = matrix.indptr
        indices = matrix.indices
        data = matrix.data
        weights = self._word_weights

        clusters = defaultdict(set)

        for doc_idx in doc_indices:
            if doc_idx < 0 or doc_idx >= self._n_docs:
                continue

            start = indptr[doc_idx]
            end = indptr[doc_idx + 1]
            if start == end:
                continue

            row_indices = indices[start:end]
            row_data = data[start:end]

            best_feature_idx = None
            best_score = 0.0

            if not excluded_indices:
                # Быстрый путь: просто максимум по всем кандидатам
                # score = tfidf * log(freq)
                # Вычисляем максимум за один проход без сортировок.
                for j in range(end - start):
                    fi = int(row_indices[j])
                    score = float(row_data[j]) * float(weights[fi])
                    if score > best_score:
                        best_score = score
                        best_feature_idx = fi
            else:
                # С исключениями: максимум по не-исключённым
                for j in range(end - start):
                    fi = int(row_indices[j])
                    if fi in excluded_indices:
                        continue
                    score = float(row_data[j]) * float(weights[fi])
                    if score > best_score:
                        best_score = score
                        best_feature_idx = fi

            if best_feature_idx is None or best_score <= 0.0:
                continue

            word = str(self._feature_names[best_feature_idx])
            clusters[word].add(int(doc_idx))

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
