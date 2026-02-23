from datastorage import DataStorage
from search_engine import SearchEngine
from data_exporter import DataExporter


data_store = DataStorage()
data_store.load_data()  # TODO: убрать постоянную загрузку
sentences = data_store.get_processed_sentences()

engine = SearchEngine(sentences, calc_word_freq=True)

most_popular_words = engine.index.getTopWordFrequency(20)
#print(most_popular_words)

words_tfidf = engine.tfidf.get_words_tfidf(w.word for w in most_popular_words)
#print(words_tfidf)

result = list(zip(most_popular_words, words_tfidf))
for elem in result:
    print(elem[0].word, elem[1])

# Пример использования DataExporter (пока не обязателен для основной логики):
# exporter = DataExporter()
# filepath = exporter.write_mean_tfidf_to_file(engine.tfidf.get_mean_tfidf())
# print(f\"TF-IDF результаты сохранены в {filepath}\")