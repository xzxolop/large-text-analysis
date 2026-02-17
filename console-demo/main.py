from datastorage import DataStorage
from invertedindex import InvertedIndex, SearchState

dataStore = DataStorage()
dataStore.load_data() # TODO: убрать постоянную загрузку

sentences = [
    "I like Python and data science",
    "Python is great for machine learning",
    "I use Python for scripting",
    "Data science uses statistics",
    "Big data and analytics",
    "Java is used in enterprise",
    "Java and Spring framework",
    "Python scripts automate tasks",
    "Machine learning with Python",
    "Statistics and probability for data science",
]

sentences = dataStore.get_processed_sentences()

index = InvertedIndex(sentences, calc_word_freq=True)
index.printWordFrequency(n = 20)