from search.search_engine import SearchEngine


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
