from datastorage import DataStorage
from invertedindex import InvertedIndex, SearchState

dataStore = DataStorage()
dataStore.load_data() # TODO: убрать постоянную загрузку
sentances = dataStore.get_processed_sentences()

index = InvertedIndex(sentances, True)
#index.get_searched_frequency()
#print("printWordFrequency. Самые популярные слова загруженные в inverted_index:")
#index.printTopWordFrequency(10)

res = index.getMeanTfidf()
print(res)

# Запись в текстовый файл
with open('tfidf_results.txt', 'w', encoding='utf-8') as f:
    for word, score in res:
        f.write(f"{word}: {score:.6f}\n")

print("Результаты сохранены в tfidf_results.txt")