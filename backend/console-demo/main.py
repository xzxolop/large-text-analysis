import time
from datastorage import DataStorage
from invertedindex import InvertedIndex

start_time = time.perf_counter()
dataStore = DataStorage()
dataStore.load_data() # TODO: убрать постоянную загрузку
end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"Время выполнения: {elapsed_time:.4f} секунд")

sentances = dataStore.get_sentances()
index = InvertedIndex(sentances)
#index.printIndex() # TODO: сделать вывод по популяронсти встреч
res = index.searchWith("russia")
index.printResult()
sent = index.getSentByIndexes(res)
print(sent)

print("end")