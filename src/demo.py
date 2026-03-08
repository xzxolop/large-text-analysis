from search.search_engine import SearchEngine
from typing import List, Tuple, Optional
from data.data_exporter import DataExporter
import math


def export_tfidf_results(engine: SearchEngine, n: int = 20, filename: str = "tfidf_results.txt") -> None:
    """
    Экспортирует TF-IDF значения топ-N слов в файл.
    
    Args:
        engine: Search engine instance.
        n: Количество топ слов для экспорта.
        filename: Имя файла для сохранения.
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

    Args:
        engine: Search engine instance.
        words: List of words to search for.
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

    Args:
        engine: Search engine instance.
        seed_word: Слово для которого ищем кластер.
        top_n: Количество результатов.
        use_npmi: Использовать Normalized PMI.
        min_freq: Минимальная частота слова.
        tfidf_range: Диапазон TF-IDF (min, max) для фильтрации.
        use_freq_weighting: Использовать комбинированный скор PMI × log(freq).
        min_score_percent: Минимальный процент от максимального score для фильтрации.
                          Например, 30.0 оставит слова с score >= 30% от максимального.
                          0.0 отключает фильтрацию по проценту.
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
        print(f"   (попробуйте уменьшить min_freq, min_score_percent или расширить tfidf_range)")
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


def calculate_score_statistics(cluster: List[Tuple[str, float]]) -> dict:
    """
    Рассчитать статистику score для кластера слов.

    Args:
        cluster: Список кортежей (слово, score).

    Returns:
        Словарь со статистикой: max, mean, median, min, count.
    """
    if not cluster:
        return {
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "count": 0,
        }

    scores = [score for _, score in cluster]
    scores_sorted = sorted(scores)
    n = len(scores)

    # Максимальный
    max_score = max(scores)

    # Минимальный
    min_score = min(scores)

    # Средний
    mean_score = sum(scores) / n

    # Медианный
    if n % 2 == 0:
        median_score = (scores_sorted[n // 2 - 1] + scores_sorted[n // 2]) / 2
    else:
        median_score = scores_sorted[n // 2]

    return {
        "max": max_score,
        "mean": mean_score,
        "median": median_score,
        "min": min_score,
        "count": n,
    }


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
    Сначала применяется фильтр по проценту от максимального score, затем сортировка по freq.

    Args:
        engine: Search engine instance.
        seed_word: Слово для которого ищем кластер.
        top_n: Количество результатов.
        use_npmi: Использовать Normalized PMI.
        min_freq: Минимальная частота слова.
        tfidf_range: Диапазон TF-IDF (min, max) для фильтрации.
        use_freq_weighting: Использовать комбинированный скор PMI × log(freq).
        min_score_percent: Минимальный процент от максимального score для фильтрации.
                          Например, 30.0 оставит слова с score >= 30% от максимального.
    """
    if not engine.is_cluster_analysis_enabled():
        print("❌ Cluster analysis is not enabled")
        return

    metric_name = "NPMI" if use_npmi else "PMI"
    print(f"\n📊 Кластер для слова: '{seed_word}' ({metric_name}, сортировка по частоте)")
    print(f"   min_score_percent={min_score_percent}%, min_freq={min_freq}")
    print("-" * 60)

    # Получаем кластер с фильтром по проценту
    cluster = engine.get_cluster_words(
        seed_word,
        top_n=top_n * 5,  # Берём с запасом, т.к. потом отфильтруем по частоте
        use_npmi=use_npmi,
        min_freq=min_freq,
        tfidf_range=tfidf_range,
        use_freq_weighting=use_freq_weighting,
        min_score_percent=min_score_percent,
    )

    if not cluster:
        print(f"   Слово '{seed_word}' не найдено в корпусе или нет ассоциаций")
        print(f"   (попробуйте уменьшить min_score_percent или min_freq)")
        return

    # Получаем частоту для каждого слова
    cluster_with_freq = []
    for word, score in cluster:
        freq = engine._cluster_analyzer.word_doc_freq.get(word.lower(), 0)
        cluster_with_freq.append((word, freq, score))

    # Сортируем по убыванию частоты
    cluster_with_freq.sort(key=lambda x: x[1], reverse=True)

    # Берём топ-N
    cluster_with_freq = cluster_with_freq[:top_n]

    # Статистика score
    stats = calculate_score_statistics([(w, s) for w, _, s in cluster_with_freq])
    
    print(f"   Статистика Score:")
    print(f"      Максимальный: {stats['max']:.4f}")
    print(f"      Средний:      {stats['mean']:.4f}")
    print(f"      Медианный:    {stats['median']:.4f}")
    print(f"      Минимальный:  {stats['min']:.4f}")
    print(f"      Количество:   {stats['count']}")

    print("-" * 60)
    print(f"   {'Слово':<20} {'Freq':>10} {'Score':>12}")
    print("-" * 60)
    for word, freq, score in cluster_with_freq:
        print(f"   {word:<20} {freq:>10} {score:>12.4f}")
