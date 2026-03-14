from search.search_engine import SearchEngine
from typing import List, Tuple, Optional
from data.data_exporter import DataExporter
import math
import time
from interface.invindex import InvIndex


def export_tfidf_results(engine: SearchEngine, n: int = 20, filename: str = "tfidf_results.txt") -> None:
    """
    Экспортирует TF-IDF значения топ-N слов в файл.
    """
    exporter = DataExporter()
    filepath = exporter.write_mean_tfidf_to_file(engine.get_top_words_with_tfidf(n=n))
    print(f"Результат сохранён в файл: {filepath}")


def tfidf_for_top_words(engine: SearchEngine, top_n: int = 20) -> None:
    most_popular_words = engine.get_top_word_frequency(top_n)
    print(f"\nMost popular words\n{most_popular_words}")

    top_tfidf = engine.get_top_words_with_tfidf(top_n)
    print(f"\nTf-idf for most popular words\n{top_tfidf}")


def search_words_sequentially(engine: SearchEngine, words: list[str]) -> None:
    """
    Performs sequential search for a list of words and prints results.
    """
    state = None
    for word in words:
        print(f"\n🔍 Searching for: {word!r}")
        state = engine.search(search_word=word, state=state)
        print(f"Found sentences:")
        state.print_matches()
        print(f"Related words (by frequency):")
        state.print_word_frequency(n=20)


def show_word_cluster(
    engine: SearchEngine,
    seed_word: str,
    top_n: int = 20,
    use_npmi: bool = False,
    min_freq: int = 1,
    tfidf_range: Optional[Tuple[float, float]] = None,
    use_freq_weighting: bool = True,
    min_score_percent: float = 0.0,
) -> None:
    """
    Показать слова кластера для заданного слова (PMI-based кластеризация).
    """
    if not engine.is_cluster_analysis_enabled():
        print("❌ Cluster analysis is not enabled")
        return

    metric_name = "NPMI" if use_npmi else "PMI"
    freq_note = " × log(freq)" if use_freq_weighting else ""
    percent_note = f", min_score_percent={min_score_percent}%" if min_score_percent > 0 else ""
    print(f"\n🔬 Кластер для слова: '{seed_word}' ({metric_name}{freq_note}{percent_note})")
    print(f"   min_freq={min_freq}, tfidf_range={tfidf_range}")
    print("-" * 60)

    cluster = engine.get_cluster_words(
        seed_word,
        top_n=top_n,
        use_npmi=use_npmi,
        min_freq=min_freq,
        tfidf_range=tfidf_range,
        use_freq_weighting=use_freq_weighting,
        min_score_percent=min_score_percent,
    )

    if not cluster:
        print(f"   Слово '{seed_word}' не найдено в корпусе или нет ассоциаций")
        return

    # Получаем частоту для каждого слова
    cluster_with_freq = []
    for word, score in cluster:
        freq = engine._cluster_analyzer.word_doc_freq.get(word.lower(), 0)
        cluster_with_freq.append((word, freq, score))

    print(f"   {'Слово':<20} {'Freq':>10} {'Score':>12}")
    print("-" * 60)
    for word, freq, score in cluster_with_freq:
        print(f"   {word:<20} {freq:>10} {score:>12.4f}")


def show_word_cluster_by_frequency(
    engine: SearchEngine,
    seed_word: str,
    top_n: int = 20,
    use_npmi: bool = False,
    min_freq: int = 1,
    tfidf_range: Optional[Tuple[float, float]] = None,
    use_freq_weighting: bool = True,
    min_score_percent: float = 30.0,
) -> None:
    """
    Показать слова кластера для заданного слова, отсортированные по убыванию частоты.
    """
    if not engine.is_cluster_analysis_enabled():
        print("❌ Cluster analysis is not enabled")
        return

    metric_name = "NPMI" if use_npmi else "PMI"
    print(f"\n📊 Кластер для слова: '{seed_word}' ({metric_name}, сортировка по частоте)")
    print(f"   min_score_percent={min_score_percent}%, min_freq={min_freq}")
    print("-" * 60)

    cluster = engine.get_cluster_words(
        seed_word,
        top_n=top_n * 5,
        use_npmi=use_npmi,
        min_freq=min_freq,
        tfidf_range=tfidf_range,
        use_freq_weighting=use_freq_weighting,
        min_score_percent=min_score_percent,
    )

    if not cluster:
        print(f"   Слово '{seed_word}' не найдено в корпусе или нет ассоциаций")
        return

    # Получаем частоту для каждого слова
    cluster_with_freq = []
    for word, score in cluster:
        freq = engine._cluster_analyzer.word_doc_freq.get(word.lower(), 0)
        cluster_with_freq.append((word, freq, score))

    # Сортируем по убыванию частоты
    cluster_with_freq.sort(key=lambda x: x[1], reverse=True)
    cluster_with_freq = cluster_with_freq[:top_n]

    print(f"   {'Слово':<20} {'Freq':>10} {'Score':>12}")
    print("-" * 60)
    for word, freq, score in cluster_with_freq:
        print(f"   {word:<20} {freq:>10} {score:>12.4f}")


# ============================================
# ДЕМО: Непересекающаяся кластеризация (Exclusive Clustering)
# ============================================

def show_exclusive_clustering(
    engine: SearchEngine,
    n: int = 1000,
    top_n: int = 20,
) -> None:
    """
    Показать результаты непересекающейся кластеризации.
    """
    clusters = engine.exclusive_clustering(n=n)
    index = InvIndex(clusters)
    top_words = index.get_top_word_frequency(top_n)
    print(f"\nTop {top_n} exclusive clusters\n{top_words}")


def show_iterative_exclusive_clustering(
    engine: SearchEngine,
    seed_words: List[str],
    top_n: int = 20,
    min_score_percent: float = 30.0,
) -> None:
    """
    Показать результаты итеративной непересекающейся кластеризации.
    """
    start_time = time.time()
    clusters = engine.iterative_exclusive_clustering(
        seed_words=seed_words,
        min_score_percent=min_score_percent,
    )
    elapsed_time = time.time() - start_time

    print(f"\n⏱️ Время выполнения: {elapsed_time:.3f} сек")

    index = InvIndex(clusters)
    top_words = index.get_top_word_frequency(top_n)
    print(f"\nTop {top_n} exclusive clusters\n{top_words}")
