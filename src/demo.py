from search.search_engine import SearchEngine
from typing import List, Tuple, Optional


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
    """
    if not engine.is_cluster_analysis_enabled():
        print("❌ Cluster analysis is not enabled")
        return
    
    metric_name = "NPMI" if use_npmi else "PMI"
    freq_note = " × log(freq)" if use_freq_weighting else ""
    print(f"\n🔬 Кластер для слова: '{seed_word}' ({metric_name}{freq_note})")
    print(f"   min_freq={min_freq}, tfidf_range={tfidf_range}")
    print("-" * 60)
    
    cluster = engine.get_cluster_words(
        seed_word, 
        top_n=top_n, 
        use_npmi=use_npmi,
        min_freq=min_freq,
        tfidf_range=tfidf_range,
        use_freq_weighting=use_freq_weighting,
    )
    
    if not cluster:
        print(f"   Слово '{seed_word}' не найдено в корпусе или нет ассоциаций")
        print(f"   (попробуйте уменьшить min_freq или расширить tfidf_range)")
        return
    
    print(f"   {'Слово':<25} {'Score':>12}")
    print("-" * 60)
    for word, score in cluster:
        print(f"   {word:<25} {score:>12.4f}")
