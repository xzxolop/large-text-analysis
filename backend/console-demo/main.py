import time
from datastorage import DataStorage
from invertedindex import InvertedIndex

start_time = time.perf_counter()
dataStore = DataStorage()
dataStore.load_data()
end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"Время выполнения: {elapsed_time:.4f} секунд")

start_time = time.perf_counter()
sentances = dataStore.get_sentances()
index = InvertedIndex(sentances[:50])
index.printIndex()
end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"Время выполнения: {elapsed_time:.4f} секунд")
