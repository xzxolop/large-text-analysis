from search.search_engine import SearchEngine


def tfidf_for_top_words(engine: SearchEngine, top_n: int = 20) -> None:
    most_popular_words = engine.get_top_word_frequency(top_n)
    print(f"\nMost popular words\n{most_popular_words}")

    top_tfidf = engine.get_top_words_with_tfidf(top_n)
    print(f"\nTf-idf for most popular words\n{top_tfidf}")
