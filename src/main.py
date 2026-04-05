from data.data_storage import DataStorage
from search.search_engine import SearchEngine
from data.data_exporter import DataExporter
from analysis.exclusive_clustererV2 import ExclusiveClustererV2
import demo
from interface.invindex import InvIndex


data_store = DataStorage()
data_store.load_data()
sentences = data_store.get_processed_sentences()

#engine = SearchEngine(sentences, calc_word_freq=True)

#demo.tfidf_for_top_words(engine, top_n=20)

#demo.export_tfidf_results(engine, n=20, filename="tfidf_results.txt")
#demo.search_words_sequentially(engine, ["russia", "china"])

# ============================================
# Пример 1: Базовый кластерный анализ слова
# ============================================
# Поиск ассоциаций для слова "russia"
# Комбинированный скор: PMI × log(freq) для подъёма частых слов
# demo.show_word_cluster(engine, "russia", top_n=20, min_freq=1, use_freq_weighting=True)

# ============================================
# Пример 2: Кластерный анализ для разных слов
# ============================================
# demo.show_word_cluster(engine, "data", top_n=20, min_freq=1, use_freq_weighting=True)
# demo.show_word_cluster(engine, "python", top_n=20, min_freq=1, use_freq_weighting=True)
# demo.show_word_cluster(engine, "ai", top_n=20, min_freq=1, use_freq_weighting=True)

# ============================================
# Пример 3: Кластерный анализ с сортировкой по частоте
# ============================================
# Top-20 слов по частоте среди релевантных (min_score_percent=30%)
# demo.show_word_cluster_by_frequency(engine, "data", top_n=20, min_freq=1, min_score_percent=30.0)
# demo.show_word_cluster_by_frequency(engine, "python", top_n=20, min_freq=1, min_score_percent=30.0)
# demo.show_word_cluster_by_frequency(engine, "russia", top_n=20, min_freq=1, min_score_percent=30.0)

# ============================================
# ДЕМО: Непересекающаяся кластеризация (TF-IDF based)
# ============================================

#n = None
#demo.show_exclusive_clustering(engine, n=n, top_n=20)
#demo.show_iterative_exclusive_clustering(engine, seed_words=["python"], top_n=20)
#demo.show_iterative_exclusive_clustering(engine, seed_words=["python", "use"], top_n=20)

print(len(sentences))
ex_clust = ExclusiveClustererV2()
clusters = ex_clust.get_clusters(sentences)
index = InvIndex(clusters)
top_words = index.get_top_word_frequency(n=20)
print(top_words)

word = "help"

sent_ind = clusters[word]

#print(sent_ind)

new_sents = []
for i in sent_ind:
    new_sents.append(sentences[i])

print(new_sents[:10])

print("sent count:", len(new_sents))
ex_clust = ExclusiveClustererV2()
clusters = ex_clust.get_clusters(new_sents, [word])
index = InvIndex(clusters)
top_words = index.get_top_word_frequency(n=20)
print(top_words)


# [deleted: 2635, thanks: 1500, removed: 977, thank: 910, performed: 827, concerns: 825, needed: 503, much: 384, luck: 364, awesome: 352, cool: 327, check: 291, great: 264, link: 251, sharing: 219, helps: 217, yes: 213, nice: 212, help: 199, interesting: 192]