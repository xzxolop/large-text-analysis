from datastorage import DataStorage
from invertedindex import InvertedIndex

dataStore = DataStorage()
dataStore.load_data() # TODO: убрать постоянную загрузку

"""
Пока работает так
index: word -> {1,2,3}
store: {1,2,3} -> ['s1', 's2', 's3']
"""

sentances = dataStore.get_processed_sentences()
index = InvertedIndex(sentances)

p = index.topOfIndex()
print("popularity:", p[:10])

# TODO: сделать вывод по популяронсти встреч
res = index.searchWith("russia")
index.printResult()
sent = dataStore.get_original_sentences_by_index(res)
print(sent)

print("end")