from datastorage import DataStorage
from invertedindex import InvertedIndex

dataStore = DataStorage()
dataStore.load_data() # TODO: убрать постоянную загрузку
sentances = dataStore.get_processed_sentences()

index = InvertedIndex(sentances, True)

most_popular_words = index.getTopWordFrequency(20)
print(most_popular_words)

words_tfidf = index.getWordsTfidf(most_popular_words)
print(words_tfidf)

result = list(zip(most_popular_words, words_tfidf))
for elem in result:
    print(elem[0].word, elem[1])