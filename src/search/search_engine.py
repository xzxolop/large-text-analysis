from typing import Iterable, List, Optional, Tuple

from core.inverted_index import InvertedIndex, SearchState, MyWord
from core.tfidf_model import TfidfModel
from analysis.cluster_analyzer import ClusterAnalyzer


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
    ) -> None:
        self._index = InvertedIndex(sentences, calc_word_freq=calc_word_freq)
        self._tfidf = TfidfModel(sentences, vectorizer=tfidf_vectorizer)
        self._cluster_analyzer = (
            ClusterAnalyzer(sentences) if enable_cluster_analysis else None
        )
        self._tfidf_cache: Optional[dict] = None

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
