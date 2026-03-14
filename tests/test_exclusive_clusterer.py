"""
Тесты для модуля непересекающейся кластеризации (ExclusiveClusterer).
"""

import pytest
import numpy as np
from scipy.sparse import csr_matrix
from analysis.exclusive_clusterer import ExclusiveClusterer


@pytest.fixture
def sample_tfidf_data():
    """Создаёт тестовые данные: TF-IDF матрицу и частоты слов."""
    tfidf_matrix = csr_matrix([
        [0.5, 0.5, 0.3, 0.0, 0.0, 0.0],
        [0.0, 0.6, 0.4, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.4, 0.5, 0.0, 0.4],
        [0.4, 0.4, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.6, 0.5],
    ])

    feature_names = np.array(["machine", "learning", "great", "data", "deep", "networks"])

    word_freqs = {
        "machine": 2,
        "learning": 3,
        "great": 2,
        "data": 2,
        "deep": 2,
        "networks": 1,
    }

    return tfidf_matrix, feature_names, word_freqs


@pytest.fixture
def clusterer(sample_tfidf_data):
    """Создание кластеризатора для тестов."""
    tfidf_matrix, feature_names, word_freqs = sample_tfidf_data
    return ExclusiveClusterer(tfidf_matrix, feature_names, word_freqs)


class TestExclusiveClustererInit:
    """Тесты инициализации ExclusiveClusterer."""

    def test_init_creates_clusterer(self, sample_tfidf_data):
        """Проверка создания кластеризатора."""
        tfidf_matrix, feature_names, word_freqs = sample_tfidf_data
        clusterer = ExclusiveClusterer(tfidf_matrix, feature_names, word_freqs)

        assert clusterer._n_docs == 5
        assert clusterer._n_features == 6
        assert len(clusterer._word_weights) == 6

    def test_word_weights_are_positive(self, clusterer):
        """Веса слов должны быть положительными."""
        assert all(w > 0 for w in clusterer._word_weights)


class TestCluster:
    """Тесты метода cluster()."""

    def test_cluster_returns_dict(self, clusterer):
        """Метод cluster должен возвращать словарь."""
        clusters = clusterer.cluster()
        assert isinstance(clusters, dict)

        for word, indices in clusters.items():
            assert isinstance(word, str)
            assert isinstance(indices, set)

    def test_cluster_all_docs_assigned(self, clusterer):
        """Все предложения должны быть назначены какому-либо кластеру."""
        clusters = clusterer.cluster(n=5)

        all_indices = set()
        for indices in clusters.values():
            all_indices.update(indices)

        assert len(all_indices) == 5

    def test_cluster_no_overlap(self, clusterer):
        """Предложения не должны пересекаться между кластерами."""
        clusters = clusterer.cluster(n=5)

        all_indices = []
        for indices in clusters.values():
            all_indices.extend(indices)

        assert len(all_indices) == len(set(all_indices))

    def test_cluster_respects_n_parameter(self, clusterer):
        """Параметр n должен ограничивать количество предложений."""
        clusters_n3 = clusterer.cluster(n=3)
        clusters_n5 = clusterer.cluster(n=5)

        total_n3 = sum(len(indices) for indices in clusters_n3.values())
        total_n5 = sum(len(indices) for indices in clusters_n5.values())

        assert total_n3 == 3
        assert total_n5 == 5

    def test_cluster_empty_n_zero(self, clusterer):
        """n=0 должен вернуть пустой результат."""
        clusters = clusterer.cluster(n=0)
        assert clusters == {}


class TestGetTopClusters:
    """Тесты метода get_top_clusters()."""

    def test_top_n_limits_results(self, clusterer):
        """top_n должен ограничивать количество кластеров."""
        top_3 = clusterer.get_top_clusters(n=3)
        top_5 = clusterer.get_top_clusters(n=5)

        assert len(top_3) <= 3
        assert len(top_5) <= 5
        assert len(top_3) <= len(top_5)

    def test_min_cluster_size_filters(self, clusterer):
        """min_cluster_size должен фильтровать маленькие кластеры."""
        top_no_filter = clusterer.get_top_clusters(n=10, min_cluster_size=1)
        top_filtered = clusterer.get_top_clusters(n=10, min_cluster_size=2)

        assert len(top_filtered) <= len(top_no_filter)

        for word, indices in top_filtered.items():
            assert len(indices) >= 2


class TestPerformance:
    """Тесты производительности (базовые)."""

    def test_large_matrix(self):
        """Кластеризация должна работать на большой матрице."""
        n_docs = 1000
        n_features = 100

        from scipy.sparse import random
        tfidf_matrix = random(n_docs, n_features, density=0.1, format="csr")
        feature_names = np.array([f"word_{i}" for i in range(n_features)])
        word_freqs = {f"word_{i}": np.random.randint(1, 100) for i in range(n_features)}

        clusterer = ExclusiveClusterer(tfidf_matrix, feature_names, word_freqs)

        clusters = clusterer.cluster()

        assert len(clusters) > 0
        total_assigned = sum(len(indices) for indices in clusters.values())
        assert total_assigned == n_docs
