from datastorage import DataStorage
from invertedindex import InvertedIndex, SearchState

#dataStore = DataStorage()
#dataStore.load_data() # TODO: убрать постоянную загрузку
#sentances = dataStore.get_processed_sentences()

sentances = [
    "apple is a fruit and apple is delicious",
    "banana is yellow like sun and banana is sweet",
    "apple and banana are both fruits",
    "fruit is healthy for body and mind",
    "sun shines bright in sky",
    "sky is blue when sun is visible",
    "mind needs knowledge like body needs food",
    "knowledge comes from books and experience",
    "books contain stories about life",
    "life is full of surprises and joy",
    "joy comes from simple things like fruit",
    "sun gives light and warmth to earth",
    "earth revolves around sun every day",
    "day starts when sun rises",
    "apple pie is tasty dessert with fruit",
    "banana bread is another sweet treat",
    "body needs exercise to stay strong",
    "mind needs books to stay sharp",
    "sky changes color at sunset",
    "life without joy is like day without sun"
]

index = InvertedIndex(sentances, True)
index.get_searched_frequency()
print("printWordFrequency. Самые популярные слова загруженные в inverted_index:")
index.printWordFrequency(10)
print("\nprintIndex")
index.printIndex()

first_searched_word = "fruit"
state = index.search(first_searched_word)
print(f"\nСамые популярные слова поиска для слова {first_searched_word}:")
state.printWordFrequency(10)
print("Поисковые слова и предожения")
state.printMatches()

second_searched_word = "apple"
state = index.search(second_searched_word, state)
print(f"\nСамые популярные слова поиска для слова {second_searched_word}:")
state.printWordFrequency(10)
print("Поисковые слова и предожения")
state.printMatches()

print("end")