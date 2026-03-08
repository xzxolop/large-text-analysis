from data.data_storage import DataStorage
from search.search_engine import SearchEngine
from data.data_exporter import DataExporter
import demo


data_store = DataStorage()
data_store.load_data()  # TODO: убрать постоянную загрузку
sentences = data_store.get_processed_sentences()

engine = SearchEngine(sentences, calc_word_freq=True)

demo.tfidf_for_top_words(engine, top_n=20)

demo.export_tfidf_results(engine, n=20, filename="tfidf_results.txt")

demo.search_words_sequentially(engine, ["russia", "china"])

# ============================================
# ДЕМО: Кластерный анализ слов (PMI-based)
# ============================================
print("\n" + "=" * 50)
print("ДЕМО: Кластерный анализ слов")
print("=" * 50)

# Пример: поиск кластера для слова "russia"
# Комбинированный скор: PMI × log(freq) для подъёма частых слов
#demo.show_word_cluster(engine, "russia", top_n=20, min_freq=1, use_freq_weighting=True)

# Пример: поиск кластера для слова "data"
demo.show_word_cluster(engine, "data", top_n=20, min_freq=1, use_freq_weighting=True)

demo.show_word_cluster(engine, "python", top_n=20, min_freq=1, use_freq_weighting=True)

#demo.show_word_cluster(engine, "ai", top_n=20, min_freq=1, use_freq_weighting=True)

# ============================================
# ДЕМО: Кластерный анализ с сортировкой по частоте
# ============================================
print("\n" + "=" * 50)
print("ДЕМО: Кластерный анализ с сортировкой по частоте")
print("=" * 50)

# Пример: top-20 слов по частоте среди релевантных (min_score_percent=30%)
demo.show_word_cluster_by_frequency(engine, "data", top_n=20, min_freq=1, min_score_percent=30.0)

demo.show_word_cluster_by_frequency(engine, "python", top_n=20, min_freq=1, min_score_percent=30.0)

demo.show_word_cluster_by_frequency(engine, "russia", top_n=20, min_freq=1, min_score_percent=30.0)