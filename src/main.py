from data.data_storage import DataStorage
from search.search_engine import SearchEngine
from data.data_exporter import DataExporter
import demo


data_store = DataStorage()
data_store.load_data()  # TODO: убрать постоянную загрузку
sentences = data_store.get_processed_sentences()

engine = SearchEngine(sentences, calc_word_freq=True)

#demo.tfidf_for_top_words(engine, top_n=20)

# Пример использования DataExporter (пока не обязателен для основной логики):
#exporter = DataExporter()
#filepath = exporter.write_mean_tfidf_to_file(engine.get_top_words_with_tfidf(n=20))

#demo.search_words_sequentially(engine, ["russia", "china"])

# Анализ наименее частых слов
# Поиск слов с конкретной частотой
#print("Поиск слов по частоте")
#words = engine.get_words_by_frequency(freq=2)
#print(f"Найдено слов с частотой 2: {len(words)}")

#tfidf = engine.get_words_tfidf(words)

# Фильтруем слова с TF-IDF != 0
#words_and_tfidf = [(w, s) for w, s in zip(words, tfidf) if s > 0]
#print(f"Слов с TF-IDF > 0: {len(words_and_tfidf)}")

#exporter.write_mean_tfidf_to_file(tfidf_list=words_and_tfidf, filename="wf1tfidf.txt")
#print(f"Результат сохранён в файл: files/wf1tfidf.txt")

# ============================================
# ДЕМО: Кластерный анализ слов (PMI-based)
# ============================================
print("\n" + "=" * 50)
print("ДЕМО: Кластерный анализ слов")
print("=" * 50)

# Пример: поиск кластера для слова "russia"
# Комбинированный скор: PMI × log(freq) для подъёма частых слов
demo.show_word_cluster(engine, "russia", top_n=20, min_freq=1, use_freq_weighting=True)

# Пример: поиск кластера для слова "data"
demo.show_word_cluster(engine, "data", top_n=20, min_freq=1, use_freq_weighting=True)

demo.show_word_cluster(engine, "python", top_n=20, min_freq=1, use_freq_weighting=True)

demo.show_word_cluster(engine, "ai", top_n=20, min_freq=1, use_freq_weighting=True)