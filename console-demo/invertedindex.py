from nltk.tokenize import word_tokenize

class WordFrequency:
    word: str
    freq: int

    def __init__(self, word, freq):
        self.word = word
        self.freq = freq

    def __eq__(self, other):  # == (equal)
        return self.word == other.word and self.freq == other.freq
    
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




class InvertedIndex:
    """
    __index         - словарь вида <string, set<int>>, где ключ - слово, а значение - множество индексов списка предложений (__sentences).\n
    __sentences:    - список уже обработанных предложений.\n
    __word_frequency    - список содержащий WordFrequency, который отражает наиболее популярные слова нашего датасета, которые сортируеются по убыванию популярности.\n
    """

    __index = dict()
    __sentences = list()
    __word_frequency = list() # нужен чтобы каждый раз не пересчитывать.

    def __init__(self, sentences: list, calc_word_freq = False):
        self.__sentences = sentences
        self.__index = self.create_index(sentences)

        if calc_word_freq:
            self.__word_frequency = self.__topOfIndex(self.__index)
    
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
            indexes = self.__search(search_word) # NOTE: это плохо. Возможна потеря данных!
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

        state.word_frequency = self.calculate_frequency(indexes) 
        return state
    
    # TODO: сделать приватным?
    def calculate_frequency(self, indexes: set):
        sent_list = self.get_sentences_by_indexes(indexes)
        index = self.create_index(sent_list)
        return self.__topOfIndex(index)
    
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
        
    def __topOfIndex(self, index: dict) -> list: # TODO: переписать это на dict
        word_frequency = []
        for key in index:
            word_freq = WordFrequency(key, len(index[key]))
            word_frequency.append(word_freq)
        word_frequency.sort(key=lambda x: x.freq, reverse=True)
        return word_frequency
