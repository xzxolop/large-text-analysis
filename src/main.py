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

# Анализ наименее частых слов
# Поиск слов с конкретной частотой
print("Поиск слов по частоте")
words = engine.get_words_by_frequency(freq=2)
print(f"Найдено слов с частотой 1: {len(words)}")

tfidf = engine.get_words_tfidf(words)
words_and_tfidf = list(zip(words, tfidf))
exporter.write_mean_tfidf_to_file(tfidf_list=words_and_tfidf, filename="wf1tfidf.txt")
print(f"Результат сохранён в файл: files/wf1tfidf.txt")