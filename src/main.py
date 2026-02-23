from datastorage import DataStorage
from search_engine import SearchEngine
from data_exporter import DataExporter


data_store = DataStorage()
data_store.load_data()  # TODO: убрать постоянную загрузку
sentences = data_store.get_processed_sentences()

engine = SearchEngine(sentences, calc_word_freq=True)

# Демонстрация работы фасадного метода get_top_word_frequency + TF-IDF через TfidfModel
most_popular_words = engine.get_top_word_frequency(20)
words_tfidf = engine.tfidf.get_words_tfidf(w.word for w in most_popular_words)

print("Top words (freq + TF-IDF):")
for word_obj, score in zip(most_popular_words, words_tfidf):
    print(word_obj.word, score)

# То же самое, что и код строчкой выше, но короче
# Демонстрация высокоуровневого метода SearchEngine.get_top_words_with_tfidf
print("\nTop words via get_top_words_with_tfidf:")
for word_obj, score in engine.get_top_words_with_tfidf(20):
    print(word_obj.word, score)

# Пример использования DataExporter (пока не обязателен для основной логики):
# exporter = DataExporter()
# filepath = exporter.write_mean_tfidf_to_file(engine.tfidf.get_mean_tfidf())
# print(f\"TF-IDF результаты сохранены в {filepath}\")