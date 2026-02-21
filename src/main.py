from datastorage import DataStorage
from invertedindex import InvertedIndex, SearchState

dataStore = DataStorage()
dataStore.load_data() # TODO: убрать постоянную загрузку
sentances = dataStore.get_processed_sentences()

index = InvertedIndex(sentances)
#index.get_searched_frequency()
#print("printWordFrequency. Самые популярные слова загруженные в inverted_index:")
#index.printTopWordFrequency(10)

res = index.getMeanTfidf()
index.writeMeanTfidfToFile(res)

print(res[:20])

dataStore.writeProcessedTextToFile()