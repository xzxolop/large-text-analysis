from nltk.tokenize import word_tokenize
import math

class MyWord:
    word: str
    freq: int
    idf: float
    tf: float
    tf_idf: float

    def __init__(self, word, freq, tf, idf, tf_idf):
        self.word = word
        self.freq = freq 
        self.tf = tf
        self.idf = idf
        self.tf_idf = tf_idf

    def __eq__(self, other):  # == (equal)
        return (
            self.word == other.word and 
            self.freq == other.freq and 
            self.tf == other.tf and
            self.idf == other.idf and
            self.tf_idf == other.tf_idf
        )
    
    def __str__(self):
        return str(f"{self.word}, {self.freq}, {self.tf_idf}")
    
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
            print(x.word, x.freq, x.tf_idf)

    def printMatches(self):
        """
        Эта функция выводит слова которые мы искали, а также предложения в которых они встретились.
        """
        print(f"{self.searched_words}, {self.searched_sentences}")




class InvertedIndex:
    """
    __index         - словарь вида <string, set<int>>, где ключ - слово, а значение - множество индексов списка предложений (__sentences).\n
    __sentences:    - список уже обработанных предложений.\n
    __word_frequency    - список содержащий WordFrequency, который отражает наиболее популярные слова нашего датасета, которые сортируеются по убыванию популярности.\n
    """

    __index = dict()
    __sentences = list()
    __word_frequency = list() # NOTE: нужен чтобы каждый раз не пересчитывать.

    def __init__(self, sentences: list, calc_word_freq = False):
        self.__sentences = sentences
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
            print(x.word, x.freq)


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
        sent_cnt = len(self.__sentences)

        for key in index: # key is word
            sent_indexes = index[key]
            word_freq = len(sent_indexes)
            idf = self.__idf(sent_cnt, word_freq)
            tf = self.__tf(key, sent_indexes)

            myWord = MyWord(word= key, 
                            freq= word_freq, 
                            tf= tf,
                            idf= idf,
                            tf_idf= tf * idf
                            )
            wordsList.append(myWord)
        wordsList.sort(key=lambda x: x.freq, reverse=True)
        return wordsList
    
    def __idf(self, sent_cnt, cnt_sent_with_word):
        return math.log(sent_cnt / cnt_sent_with_word)

    def __tf(self, word: str, sent_indexes: set[int]) -> int:
        tf = 0
        for i in sent_indexes:
            sent = self.__sentences[i]
            words = word_tokenize(sent)
            tf += words.count(word)
        return tf

