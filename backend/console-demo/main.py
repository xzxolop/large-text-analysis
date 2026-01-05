import time
from datastorage import DataStorage

start_time = time.perf_counter()
dataStore = DataStorage()
dataStore.load_data()
end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"Время выполнения: {elapsed_time:.4f} секунд")

