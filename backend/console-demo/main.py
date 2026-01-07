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

res = index.searchWith("data")
index.printWordFrequency(20)

print("")

res = index.searchWith("use")
index.printWordFrequency(20)

print("")

res = index.searchWith("get")
index.printWordFrequency(20)
#index.printResult()
#sent = dataStore.get_original_sentences_by_index(res)
#print(sent)

print("end")