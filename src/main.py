from data.data_storage import DataStorage
from search.search_engine import SearchEngine
from data.data_exporter import DataExporter
from demo import tfidf_for_top_words


data_store = DataStorage()
data_store.load_data()  # TODO: убрать постоянную загрузку
sentences = data_store.get_processed_sentences()

engine = SearchEngine(sentences, calc_word_freq=True)

tfidf_for_top_words(engine, top_n=20)

# Пример использования DataExporter (пока не обязателен для основной логики):
exporter = DataExporter()
filepath = exporter.write_mean_tfidf_to_file(engine.get_top_words_with_tfidf(n=20))
