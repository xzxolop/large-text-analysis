from datastorage import DataStorage
from invertedindex import InvertedIndex, SearchState

dataStore = DataStorage()
dataStore.load_data() # TODO: убрать постоянную загрузку
#sentences = dataStore.get_processed_sentences()
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


index = InvertedIndex(sentences, calc_word_freq=True)

all_sentences = set(range(len(sentences)))
tree = index.build_cluster_tree(all_sentences, min_size=1)

index.print_cluster_tree(tree)

dot = index.visualize_cluster_tree_graphviz(tree)
dot.render("cluster_tree", format="png", cleanup=True)

print("end")