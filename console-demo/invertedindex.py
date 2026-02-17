from nltk.tokenize import word_tokenize
import math
from graphviz import Digraph

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans



class MyWord:
    word: str
    freq: int
    score: float

    def __init__(self, word, freq, score):
        self.word = word
        self.freq = freq
        self.score = score # TF-IDF * balance

    def __eq__(self, other):  # == (equal)
        return self.word == other.word and self.freq == other.freq and self.score == other.score
    
    def __str__(self):
        return str(f"{self.word}: {self.freq}")
    
class SearchState:
    """
    Docstring for SearchState
    
    searched_words        - слова по которым прошел поиск.\n
    searched_sentences    - предложения, в которых слова встретились.\n
    searched_frequency    - список содержащий WordFrequency, который отражает наиболее популярные слова, которые сортируеются по убыванию популярности.\n
    """

    # TODO: стоит ли сделать приватными?
    searched_words = set()
    searched_sentences = set()
    word_frequency = list()

    def clearState(self):
        """
        Очистить состояние поиска.
        """
        self.searched_words.clear()
        self.searched_sentences.clear()
        self.word_frequency.clear()

    def printWordFrequency(self, n = None):
        """
        Выводит какие слова и насколько часто встречаются с поисковым(и) словом(ами)
        """
        size = len(self.word_frequency)
        if (n == None or size < n):
            n = size
        
        for x in self.word_frequency[:n]:
            print(x.word, x.freq)

    def printMatches(self):
        """
        Эта функция выводит слова которые мы искали, а также предложения в которых они встретились.
        """
        print(f"{self.searched_words}, {self.searched_sentences}")

class ClusterNode:
    def __init__(self, split_word=None, sent_indexes=None):
        self.split_word = split_word
        self.sent_indexes = sent_indexes or set()
        self.left = None
        self.right = None

class InvertedIndex:
    """
    __index         - словарь вида <string, set<int>>, где ключ - слово, а значение - множество индексов списка предложений (__sentences).\n
    __sentences:    - список уже обработанных предложений.\n
    __word_frequency    - список содержащий WordFrequency, который отражает наиболее популярные слова нашего датасета, которые сортируеются по убыванию популярности.\n
    """

    __index = dict()
    __sentences = list()
    __word_frequency = list() # NOTE: нужен чтобы каждый раз не пересчитывать.
    
    __total_docs: int
    # ===== TF-IDF + KMeans =====
    __tfidf_vectorizer = None
    __tfidf_matrix = None
    __kmeans_model = None
    __kmeans_labels = None


    def __init__(self, sentences: list, calc_word_freq=False, use_fast_tokenizer=False):
        self.__sentences = sentences
        self.__total_docs = len(sentences)
        self.__use_fast_tokenizer = use_fast_tokenizer
        self.__index = self.create_index(sentences)

        if calc_word_freq:
            self.__word_frequency = self.__convertIndexToList(self.__index)

    
    def create_index(self, sentences: list) -> dict:
        index = dict()
        for i in range(len(sentences)):
            sent = sentences[i]
            words = word_tokenize(sent)
            for word in words:
                if word in index:
                    index[word].add(i)
                else:
                    s = {i}
                    index[word] = s
        return index
    
    def search(self, search_word: str, state = SearchState()) -> SearchState:
        """
        Функция для последовательного поиска слов (с памятью).
        Возвращает множество индексов предложений в которых встречалось поисковое слово.
        """

        if search_word in state.searched_words:
            return state
        
        indexes = set()

        if len(state.searched_words) == 0:
            indexes = self.__search(search_word)
            state.searched_words.add(search_word)
            state.searched_sentences = indexes
        else:
            for x in state.searched_sentences:
                sent = self.__sentences[x]
                words = word_tokenize(sent)
                if search_word in words:
                    indexes.add(x)

            state.searched_sentences = indexes
            state.searched_words.add(search_word)

        state.word_frequency = self.__calculate_frequency(indexes) 
        return state
    
    def __calculate_frequency(self, indexes: set) -> list:
        """
        Эта функция нужна, чтобы для поискового слова (например python) найти слова которые чаще всего с ним встречаются.
        На вход принимает номера предожений в которых встречается поисковое слово.
        """
        sent_list = self.get_sentences_by_indexes(indexes)
        index = self.create_index(sent_list)
        return self.__convertIndexToList(index)
    
    # NOTE: Дублирование кода с datastorage
    def get_sentences_by_indexes(self, indexes: set) -> list:
        sent_list = []
        for i in indexes:
            sent_list.append(self.__sentences[i])
        return sent_list
    
    def get_searched_frequency(self):
        """
        Возвращает список частот слов по итогу загрузки датасета и если при загрузке был указан флаг calc_word_freq = False.
        Если поиска не было, то пустой список.
        """
        return self.__word_frequency.copy()

    # TODO: сделать вывод по популяронсти встреч
    def printIndex(self, n = None):
        """
        Печатает весь индекс как структуру данных.
        """
        if n == None or n > len(self.__index):
            print(self.__index)
        else:
            firstOfN = list(self.__index.items())[:n]
            for key, value in firstOfN:
                print(f"{key}: {value}")
    
    def printWordFrequency(self, n = None):
        """
        Выводит наиболее популярные слова среди загруженных предложений.
        """
        size = len(self.__word_frequency)
        if (n == None or size < n):
            n = size
        
        for x in self.__word_frequency[:n]:
            print(f"{x.word} | freq={x.freq} | score={x.score:.4f}")

    def get_top_words_for_cluster(self, sent_indexes: set, top_n=5):
        index = self.create_index(self.get_sentences_by_indexes(sent_indexes))
        words = self.__convertIndexToList(index)
        words.sort(key=lambda x: x.score, reverse=True)
        return words[:top_n]
    
    def choose_split_word(self, sent_indexes: set):
        index = self.create_index(self.get_sentences_by_indexes(sent_indexes))
        words = self.__convertIndexToList(index)

        best_word = None
        best_balance = 0.0

        for w in words:
            df = len(self.__index.get(w.word, []))
            left = sent_indexes & self.__index.get(w.word, set())
            right = sent_indexes - left

            if len(left) == 0 or len(right) == 0:
                continue

            balance = 1.0 - abs(len(left) / len(sent_indexes) - 0.5)
            score = w.score * balance

            if score > best_balance:
                best_balance = score
                best_word = w.word

        return best_word

    def build_cluster_tree(self, sent_indexes: set, min_size=30):
        node = ClusterNode(sent_indexes=sent_indexes)

        if len(sent_indexes) <= min_size:
            return node

        split_word = self.choose_split_word(sent_indexes)
        if split_word is None:
            return node

        node.split_word = split_word
        left = sent_indexes & self.__index.get(split_word, set())
        right = sent_indexes - left

        if len(left) == 0 or len(right) == 0:
            return node

        node.left = self.build_cluster_tree(left, min_size)
        node.right = self.build_cluster_tree(right, min_size)
        return node
    
    def print_cluster_tree(self, node, depth=0):
        indent = "  " * depth

        print(f"{indent}Cluster size: {len(node.sent_indexes)}")

        top_words = self.get_top_words_for_cluster(node.sent_indexes)
        if top_words:
            print(f"{indent}Top words: {', '.join(w.word for w in top_words)}")

        if node.split_word:
            print(f"{indent}Split by: '{node.split_word}'")

        if node.left:
            print(f"{indent}Left:")
            self.print_cluster_tree(node.left, depth + 1)

        if node.right:
            print(f"{indent}Right:")
            self.print_cluster_tree(node.right, depth + 1)



    def __search(self, search_word) -> set:
        if search_word not in self.__index:
            return {}
        else:
            return self.__index[search_word]
        
    def __convertIndexToList(self, index: dict) -> list:
        """
        Преобразует инвертированный индекс в список, отсоритрованный по популярности встреч слова в предложениях.
        """
        wordsList = []
        for word in index:
            word_freq = len(index[word])
            
            if word_freq == 0:
                continue

            idf = self.__idf(word_freq)
            balance = self.__balance_score(word_freq)
            score = word_freq * idf * balance

            myWord = MyWord(word, word_freq, score)
            wordsList.append(myWord)
        wordsList.sort(key=lambda x: x.freq, reverse=True)
        return wordsList

    def __idf(self, df: int) -> float:
        return math.log((self.__total_docs + 1) / (df + 1))

    def __balance_score(self, df: int) -> float:
        p = df / self.__total_docs
        return 1.0 - abs(p - 0.5)

    def visualize_cluster_tree_graphviz(self, root):
        dot = Digraph("ClusterTree")
        dot.attr(rankdir="TB", nodesep="0.6", ranksep="0.8")

        node_id = 0

        def visit(node):
            nonlocal node_id
            current_id = node_id
            node_id += 1

            if node.split_word:
                label = f"{node.split_word}\nsize={len(node.sent_indexes)}"
            else:
                label = f"LEAF\nsize={len(node.sent_indexes)}"

            dot.node(str(current_id), label, shape="ellipse")

            if node.left:
                left_id = visit(node.left)
                dot.edge(str(current_id), str(left_id), label="yes")

            if node.right:
                right_id = visit(node.right)
                dot.edge(str(current_id), str(right_id), label="no")

            return current_id

        visit(root)
        return dot


    def build_kmeans_clusters(self, n_clusters=20, max_features=5000):
        """
        TF-IDF + MiniBatchKMeans кластеризация.
        Подходит для 100k+ предложений.
        """

        print("\nBuilding TF-IDF matrix...")

        self.__tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features
        )

        self.__tfidf_matrix = self.__tfidf_vectorizer.fit_transform(self.__sentences)

        print("TF-IDF shape:", self.__tfidf_matrix.shape)

        print("Clustering with MiniBatchKMeans...")

        self.__kmeans_model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=1000,
            n_init=10
        )

        self.__kmeans_labels = self.__kmeans_model.fit_predict(self.__tfidf_matrix)

        print("Clustering finished.\n")

    def get_kmeans_top_words(self, cluster_id: int, top_n=10):
        if self.__kmeans_model is None:
            return []

        centroid = self.__kmeans_model.cluster_centers_[cluster_id]
        feature_names = self.__tfidf_vectorizer.get_feature_names_out()

        top_indices = centroid.argsort()[-top_n:][::-1]
        return [feature_names[i] for i in top_indices]

    def get_kmeans_cluster_sentences(self, cluster_id: int, limit=None):
        if self.__kmeans_labels is None:
            return []

        indexes = np.where(self.__kmeans_labels == cluster_id)[0]

        if limit:
            indexes = indexes[:limit]

        return [self.__sentences[i] for i in indexes]

    def print_kmeans_clusters(self, top_n=10, show_examples=False, example_limit=3):
        if self.__kmeans_model is None:
            print("KMeans not built yet.")
            return

        print("=" * 60)
        print("KMEANS CLUSTER SUMMARY")
        print("=" * 60)

        for cluster_id in range(self.__kmeans_model.n_clusters):

            size = np.sum(self.__kmeans_labels == cluster_id)
            top_words = self.get_kmeans_top_words(cluster_id, top_n)

            print(f"\nCluster {cluster_id}")
            print(f"Size: {size}")
            print("Top words:", ", ".join(top_words))

            if show_examples:
                print("Examples:")
                examples = self.get_kmeans_cluster_sentences(cluster_id, example_limit)
                for s in examples:
                    print("  -", s)

        print("\nDone.\n")
