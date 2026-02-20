from datastorage import DataStorage
from invertedindex import InvertedIndex, SearchState

dataStore = DataStorage()
dataStore.load_data() # TODO: убрать постоянную загрузку
sentances = dataStore.get_processed_sentences()

index = InvertedIndex(sentances, True)
index.get_searched_frequency()
print("printWordFrequency. Самые популярные слова загруженные в inverted_index:")
index.printWordFrequency(10)
#print("\nprintIndex")
#index.printIndex()

first_searched_word = "data"
state = index.search(first_searched_word)
print(f"\nСамые популярные слова поиска для слова {first_searched_word}:")
state.printWordFrequency(10)
#print("Поисковые слова и предожения")
#state.printMatches()

second_searched_word = "python"
state = index.search(second_searched_word, state)
print(f"\nСамые популярные слова поиска для слова {second_searched_word}:")
state.printWordFrequency(10)
#print("Поисковые слова и предожения")
#state.printMatches()

print("end")