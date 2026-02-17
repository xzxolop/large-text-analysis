from datastorage import DataStorage
from invertedindex import InvertedIndex, SearchState

dataStore = DataStorage()
dataStore.load_data() # TODO: убрать постоянную загрузку

sentences = dataStore.get_processed_sentences()

# Для больших корпусов (100k+): быстрая токенизация и крупные листья дерева
use_fast = len(sentences) > 20_000
min_leaf = max(30, len(sentences) // 100) if len(sentences) > 10_000 else 1

index = InvertedIndex(sentences, calc_word_freq=True, use_fast_tokenizer=use_fast)

print("\n===== KMEANS CLUSTERING =====")

n_clusters = 20 if len(sentences) > 10000 else 3

index.build_kmeans_clusters(
    n_clusters=n_clusters,
    max_features=5000
)

index.print_kmeans_clusters(
    top_n=8,
    show_examples=True
)
