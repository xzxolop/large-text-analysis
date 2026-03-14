from typing import Iterable, List, Literal, Optional, Tuple, Union, Dict, Set

from nltk import word_tokenize
import math
import numpy as np
from collections import defaultdict

from core.inverted_index import InvertedIndex, SearchState, MyWord
from core.tfidf_model import TfidfModel
from analysis.cluster_analyzer import ClusterAnalyzer
from analysis.exclusive_clusterer import ExclusiveClusterer

class SearchEngine:
    """
    Высокоуровневый фасад для работы с поиском:
    скрывает внутри себя инвертированный индекс, TF-IDF модель и кластерный анализ.
    """

    def __init__(
        self,
        sentences: List[str],
        calc_word_freq: bool = False,
        tfidf_vectorizer=None,
        enable_cluster_analysis: bool = True,
        enable_exclusive_clustering: bool = True,
    ) -> None:
        self._index = InvertedIndex(sentences, calc_word_freq=calc_word_freq)
        self._tfidf = TfidfModel(sentences, vectorizer=tfidf_vectorizer)
        self._cluster_analyzer = (
            ClusterAnalyzer(sentences) if enable_cluster_analysis else None
        )
        self._tfidf_cache: Optional[dict] = None
        self._sentences = sentences

        # Инициализируем ExclusiveClusterer если включено
        self._exclusive_clusterer = None
        if enable_exclusive_clustering:
            feature_names = self._tfidf._vectorizer.get_feature_names_out()
            word_freqs = {
                word: self._index.get_word_freq(word)
                for word in feature_names
            }
            self._exclusive_clusterer = ExclusiveClusterer(
                tfidf_matrix=self._tfidf._matrix,
                feature_names=feature_names,
                word_freqs=word_freqs,
            )

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

    def get_words_by_frequency(self, freq: int) -> list[str]:
        """Вернуть все слова с указанной частотой."""
        return self._index.get_words_by_frequency(freq)

    def print_top_word_frequency(self, n: int | None = None) -> None:
        """Печатает top-N самых частых слов по индексу."""
        self._index.print_top_word_frequency(n)

    def get_word_freq(self, word: str):
        return self._index.get_word_freq(word)

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

# Функции кластерного анализа (ClusterAnalyzer)

    def get_cluster_words(
        self,
        seed_word: str,
        top_n: int = 20,
        min_pmi: float = 0.0,
        filter_pos: bool = True,
        use_npmi: bool = False,
        min_freq: int = 1,
        tfidf_range: Optional[Tuple[float, float]] = None,
        use_freq_weighting: bool = True,
        min_score_percent: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        Получить слова кластера для заданного слова (PMI-based).

        Args:
            seed_word: Исходное слово (запрос пользователя).
            top_n: Количество возвращаемых результатов.
            min_pmi: Минимальный PMI для фильтрации.
            filter_pos: Фильтровать по частям речи (существительные, глаголы, прилагательные).
            use_npmi: Использовать Normalized PMI вместо обычного.
            min_freq: Минимальная частота слова (в скольких документах встречается).
            tfidf_range: Диапазон TF-IDF (min, max) для фильтрации.
            use_freq_weighting: Если True, использовать комбинированный скор PMI × log(freq).
            min_score_percent: Минимальный процент от максимального score для фильтрации.
                              Например, 30.0 оставит слова с score >= 30% от максимального.
                              0.0 отключает фильтрацию по проценту.

        Returns:
            Список кортежей (слово, score), отсортированный по убыванию score.

        Raises:
            RuntimeError: Если кластерный анализ не включён.
        """
        if self._cluster_analyzer is None:
            raise RuntimeError("Cluster analysis is not enabled. "
                               "Set enable_cluster_analysis=True when creating SearchEngine.")

        # Получаем TF-IDF для всех слов для фильтрации (с кэшированием)
        word_tfidf_scores = None
        if tfidf_range is not None:
            if self._tfidf_cache is None:
                all_words = list(self._cluster_analyzer.word_doc_freq.keys())
                tfidf_values = self._tfidf.get_words_tfidf(all_words)
                self._tfidf_cache = dict(zip(all_words, tfidf_values))
            word_tfidf_scores = self._tfidf_cache

        return self._cluster_analyzer.get_cluster_words(
            seed_word, top_n, min_pmi, filter_pos, use_npmi,
            min_freq, tfidf_range, word_tfidf_scores, use_freq_weighting, min_score_percent
        )
    
    def get_cluster_with_frequency(
        self,
        seed_word: str,
        top_n: int = 20,
        min_pmi: float = 0.0,
        filter_pos: bool = True,
    ) -> List[Tuple[str, int, float]]:
        """
        Получить слова кластера с частотой встречаемости.
        
        Args:
            seed_word: Исходное слово.
            top_n: Количество результатов.
            min_pmi: Минимальный PMI.
            filter_pos: Фильтровать по POS.
            
        Returns:
            Список кортежей (слово, частота, pmi_score).
        """
        if self._cluster_analyzer is None:
            raise RuntimeError("Cluster analysis is not enabled.")
        
        return self._cluster_analyzer.get_cluster_with_frequency(
            seed_word, top_n, min_pmi, filter_pos
        )
    
    def is_cluster_analysis_enabled(self) -> bool:
        """Проверить, включён ли кластерный анализ."""
        return self._cluster_analyzer is not None

    def get_cluster_sentences(
        self,
        seed_word: str,
        top_n: int = 20,
        min_pmi: float = 0.0,
        filter_pos: bool = True,
        use_npmi: bool = False,
        min_freq: int = 1,
        tfidf_range: Optional[Tuple[float, float]] = None,
        use_freq_weighting: bool = True,
        min_score_percent: float = 0.0,
    ) -> Tuple[List[Tuple[str, int, float]], List[int]]:
        """
        Получить слова кластера и индексы предложений с seed_word.

        Args:
            seed_word: Исходное слово (запрос пользователя).
            top_n: Количество возвращаемых результатов.
            min_pmi: Минимальный PMI для фильтрации.
            filter_pos: Фильтровать по частям речи (существительные, глаголы, прилагательные).
            use_npmi: Использовать Normalized PMI вместо обычного.
            min_freq: Минимальная частота слова (в скольких документах встречается).
            tfidf_range: Диапазон TF-IDF (min, max) для фильтрации.
            use_freq_weighting: Если True, использовать комбинированный скор PMI × log(freq).
            min_score_percent: Минимальный процент от максимального score для фильтрации.
                              Например, 30.0 оставит слова с score >= 30% от максимального.
                              0.0 отключает фильтрацию по проценту.

        Returns:
            Кортеж из двух элементов:
            - Список кортежей (слово, частота, score), отсортированный по убыванию score.
            - Список индексов предложений, содержащих seed_word.

        Raises:
            RuntimeError: Если кластерный анализ не включён.
        """
        # Получаем кластер
        cluster = self.get_cluster_words(
            seed_word, top_n, min_pmi, filter_pos, use_npmi,
            min_freq, tfidf_range, word_tfidf_scores=None,
            use_freq_weighting=use_freq_weighting, min_score_percent=min_score_percent
        )

        # Получаем индексы предложений через InvertedIndex
        state = self.search(seed_word)
        sentence_indexes = list(state.searched_sentences)

        # Добавляем частоту к каждому слову кластера
        cluster_with_freq = [
            (word, self._cluster_analyzer.word_doc_freq.get(word.lower(), 0), score)
            for word, score in cluster
        ]

        return cluster_with_freq, sentence_indexes

# ========== Непересекающаяся кластеризация (ExclusiveClusterer) ==========

    def exclusive_clustering(
        self,
        n: Optional[int] = None,
        metric: Literal["tfidf_logfreq", "tfidf", "tfidf_div_logfreq"] = "tfidf_logfreq",
    ) -> Dict[str, Set[int]]:
        """
        Непересекающаяся кластеризация: каждое предложение получает одно релевантное слово.

        Использует векторизованные операции для высокой производительности.
        Для 100 000 предложений работает за ~1-5 секунд.

        Args:
            n: Количество предложений для обработки. Если None — все предложения.
            metric: Метрика релевантности:
                - "tfidf_logfreq": TF-IDF × log(freq) — баланс важности и частоты (по умолчанию)
                - "tfidf": чистый TF-IDF — только важность слова в предложении
                - "tfidf_div_logfreq": TF-IDF / log(freq+1) — подъём редких слов

        Returns:
            Словарь {слово: множество индексов предложений}.
            Каждое предложение принадлежит только одному кластеру (одному слову).

        Raises:
            RuntimeError: Если exclusive clustering не включён.

        Пример:
            >>> index = engine.exclusive_clustering(n=1000)
            >>> # {'data': {0, 5, 23}, 'python': {1, 12}, ...}
        """
        if self._exclusive_clusterer is None:
            raise RuntimeError(
                "Exclusive clustering is not enabled. "
                "Set enable_exclusive_clustering=True when creating SearchEngine."
            )

        return self._exclusive_clusterer.cluster(n=n, metric=metric)

    def exclusive_clustering_with_stats(
        self,
        n: Optional[int] = None,
        metric: Literal["tfidf_logfreq", "tfidf", "tfidf_div_logfreq"] = "tfidf_logfreq",
    ) -> Tuple[Dict[str, Set[int]], Dict[str, dict]]:
        """
        Непересекающаяся кластеризация со статистикой по кластерам.

        Args:
            n: Количество предложений для обработки.
            metric: Метрика релевантности.

        Returns:
            Кортеж из:
            - Словарь {слово: множество индексов предложений}
            - Словарь {слово: статистика} со статистикой по каждому кластеру

        Raises:
            RuntimeError: Если exclusive clustering не включён.
        """
        if self._exclusive_clusterer is None:
            raise RuntimeError("Exclusive clustering is not enabled.")

        clusters = self._exclusive_clusterer.cluster(n=n, metric=metric)
        stats = self._exclusive_clusterer.get_cluster_stats(clusters)

        return clusters, stats

    def get_top_exclusive_clusters(
        self,
        top_n: int = 20,
        min_cluster_size: int = 1,
        metric: Literal["tfidf_logfreq", "tfidf", "tfidf_div_logfreq"] = "tfidf_logfreq",
    ) -> Dict[str, Set[int]]:
        """
        Получить топ-N крупнейших кластеров.

        Args:
            top_n: Количество кластеров для возврата.
            min_cluster_size: Минимальный размер кластера для фильтрации.
            metric: Метрика релевантности.

        Returns:
            Словарь {слово: множество индексов} для топ-N кластеров.

        Raises:
            RuntimeError: Если exclusive clustering не включён.
        """
        if self._exclusive_clusterer is None:
            raise RuntimeError("Exclusive clustering is not enabled.")

        return self._exclusive_clusterer.get_top_clusters(
            n=top_n,
            min_cluster_size=min_cluster_size,
        )

    def exclusive_clustering_legacy(self, n: int) -> dict:
        """
        Устаревшая версия exclusive_clustering (медленная, на циклах).

        Сохранена для сравнения производительности и отладки.
        Используйте exclusive_clustering() для production.

        Args:
            n: Количество предложений для обработки.

        Returns:
            Словарь {слово: множество индексов предложений}.
        """
        index = dict()
        for i in range(len(self._sentences[:n])):
            sent = self._sentences[i]
            word = self._best_word(sent, i)
            if word in index:
                index[word].add(i)
            else:
                s = {i}
                index[word] = s
        return index

    def _best_word(self, sent, idx):
        """
        Устаревший метод для поиска лучшего слова в предложении.
        Используется только в exclusive_clustering_legacy.
        """
        words = word_tokenize(sent)
        best_word = ""
        max_score = 0
        for w in words:
            tf_idf = self._tfidf.get_word_tfidf_in_sentence(w, idx)
            freq = self._index.get_word_freq(w)
            score = tf_idf * math.log(freq + 1)
            if max_score < score:
                max_score = score
                best_word = w
        return best_word
        

        