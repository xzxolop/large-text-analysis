from data.data_storage import DataStorage
from search.search_engine import SearchEngine
from data.data_exporter import DataExporter
import demo


data_store = DataStorage()
data_store.load_data()  # TODO: убрать постоянную загрузку
sentences = data_store.get_processed_sentences()

engine = SearchEngine(sentences, calc_word_freq=True)

demo.tfidf_for_top_words(engine, top_n=20)

# Пример использования DataExporter (пока не обязателен для основной логики):
exporter = DataExporter()
filepath = exporter.write_mean_tfidf_to_file(engine.get_top_words_with_tfidf(n=20))

demo.search_words_sequentially(engine, ["russia", "china"])

# Анализ наименее частых слов:
print("\n" + "="*50)
print("Анализ наименее частых слов")
print("="*50)

least_frequent = engine.get_least_frequent_words(n=20)
print(f"\n20 наименее частых слов:\n{least_frequent}")

# Поиск слов с конкретной частотой
print("Поиск слов по частоте")
words = engine.get_words_by_frequency(freq=1)
print(words)