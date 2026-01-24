from datastorage import DataStorage
from invertedindex import InvertedIndex, SearchState

#dataStore = DataStorage()
#dataStore.load_data() # TODO: убрать постоянную загрузку
#sentances = dataStore.get_processed_sentences()

sentances = [
    "my name is ononim",
    "ononim is very rare name",
    "i play in computer",
    "computer is complex system",
    "computer is not ononim"
]

index = InvertedIndex(sentances, True)
index.get_searched_frequency()
print("printWordFrequency. Самые популярные слова загруженные в inverted_index:")
index.printWordFrequency()
print("\nprintIndex")
index.printIndex()

first_searched_word = "is"
state = index.search(first_searched_word)
print(f"\nСамые популярные слова поиска для слова {first_searched_word}:")
state.printWordFrequency()
print("Поисковые слова и предожения")
state.printMatches()

second_searched_word = "ononim"
state = index.search(second_searched_word, state)
print(f"\nСамые популярные слова поиска для слова {second_searched_word}:")
state.printWordFrequency()
print("Поисковые слова и предожения")
state.printMatches()

print("end")