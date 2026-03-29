from sklearn.feature_extraction.text import TfidfVectorizer
from interface.invindex import InvIndex
from nltk import word_tokenize
from collections import Counter
import math
import numpy as np


class ExclusiveClustererV2:
    """
    Непересекающаяся кластеризация: каждое предложение получает одно релевантное слово.

    Использует метрику TF-IDF × log(freq) для выбора наиболее релевантного слова.
    """

    def __init__(self):
        """
        Инициализация кластеризатора.

        TF-IDF векторизатор создаётся один раз и переиспользуется для всех вызовов.
        """
        self._vectorizer = TfidfVectorizer()

    def get_clusters(self, sentences: list[str]) -> InvIndex:
        """
        Построить инвертированный индекс кластеров для заданных предложений.

        Для каждого предложения выбирается одно слово с максимальным TF-IDF.
        Если предложение не содержит слов — оно пропускается.

        Args:
            sentences: Список предложений для кластеризации.

        Returns:
            InvIndex: {слово: множество индексов предложений}
        """
        if not sentences:
            return InvIndex({})

        # Векторизация предложений
        matrix = self._vectorizer.fit_transform(sentences)
        feature_names = self._vectorizer.get_feature_names_out()

        # Подсчёт частоты слов (в скольких предложениях встречается)
        word_doc_freq: Counter = Counter()
        for sent in sentences:
            words = word_tokenize(sent.lower())
            unique_words = set(words)
            for w in unique_words:
                word_doc_freq[w] += 1

        # Кэшируем log(freq) для всех слов
        word_log_freq: dict[str, float] = {}
        for word, freq in word_doc_freq.items():
            word_log_freq[word] = math.log2(freq) if freq > 0 else 0.0

        clusters: dict[str, set[int]] = {}

        # Работаем напрямую со sparse матрицей — без toarray() и flatten()
        for i in range(len(sentences)):
            # Получаем ненулевые элементы строки (эффективно для sparse)
            row = matrix.getrow(i)
            indices = row.indices
            data = row.data

            # Пропускаем пустые предложения
            if len(indices) == 0:
                continue

            # Находим индекс максимального TF-IDF без полной сортировки
            max_idx_in_row = np.argmax(data)
            feature_idx = indices[max_idx_in_row]
            word = feature_names[feature_idx]

            # Добавляем предложение в кластер слова
            if word in clusters:
                clusters[word].add(i)
            else:
                clusters[word] = {i}

        return clusters

    def get_clusters_with_scores(
        self, sentences: list[str]
    ) -> dict[str, set[float]]:
        """
        Построить кластеры с сохранением скоров для каждого предложения.

        Args:
            sentences: Список предложений для кластеризации.

        Returns:
            {слово: множество скоров (TF-IDF × log(freq))}
        """
        if not sentences:
            return {}

        # Векторизация предложений
        matrix = self._vectorizer.fit_transform(sentences)
        feature_names = self._vectorizer.get_feature_names_out()

        # Подсчёт частоты слов (в скольких предложениях встречается)
        word_doc_freq: Counter = Counter()
        for sent in sentences:
            words = word_tokenize(sent.lower())
            unique_words = set(words)
            for w in unique_words:
                word_doc_freq[w] += 1

        # Кэшируем log(freq) для всех слов
        word_log_freq: dict[str, float] = {}
        for word, freq in word_doc_freq.items():
            word_log_freq[word] = math.log2(freq) if freq > 0 else 0.0

        clusters: dict[str, set[float]] = {}

        for i in range(len(sentences)):
            row = matrix.getrow(i)
            indices = row.indices
            data = row.data

            if len(indices) == 0:
                continue

            # Находим слово с максимальным TF-IDF
            max_idx_in_row = np.argmax(data)
            feature_idx = indices[max_idx_in_row]
            word = feature_names[feature_idx]
            tf_idf = data[max_idx_in_row]

            # Рассчитываем скор
            log_freq = word_log_freq.get(word, 0.0)
            score = tf_idf * log_freq

            if word in clusters:
                clusters[word].add(score)
            else:
                clusters[word] = {score}

        return clusters
