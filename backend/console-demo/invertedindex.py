from nltk.tokenize import word_tokenize

class WordFrequency:
    word: str
    freq: int

    def __init__(self, word, freq):
        self.word = word
        self.freq = freq

class InvertedIndex:
    """
    __index         - словарь вида <string, set<int>>, где ключ - слово, а значение - множество индексов списка предложений (__sentences).\n
    __sentences:    - список уже обработанных предложений.\n

    __searched_words        - слова по которым прошел поиск.\n
    __searched_sentences    - предложения, в которых слова встретились.\n
    __searched_frequency    - список WordFrequency, который отражает наиболее популярные слова, который сортируется по убыванию.\n
    """

    __index = dict()
    __sentences: list

    __searched_words = set()
    __searched_sentences = set()
    __searched_frequency = list()

    def __init__(self, sentences: list):
        self.__sentences = sentences
        self.__index = self.create_index(sentences)
    
    def create_index(self, sentences: list) -> dict:
        index = dict()
        for i in range(len(sentences)):
            sent = sentences[i]
            words = word_tokenize(sent)
            for word in words:
                if word in index:
                    index[word].add(i)
                else:
                    s = set()
                    s.add(i)
                    index[word] = s
        return index
    
    def searchWith(self, search_word) -> set:
        """
        Функция для последовательного поиска слов (с памятью)
        """

        if search_word in self.__searched_words:
            return {} # TODO: мб кинуть ошибку
        
        s = set()


        if len(self.__searched_words) == 0:
            s = self.__search(search_word) # NOTE: это плохо. Возможна потеря данных!
            self.__searched_words.add(search_word)
            self.__searched_sentences = s.copy()
        else:
            for x in self.__searched_sentences:
                sent = self.__sentences[x]
                words = word_tokenize(sent)
                if search_word in words:
                    s.add(x)

            self.__searched_sentences = s.copy()
            self.__searched_words.add(search_word) # Нужно сделать так, чтобы при добавлении set не фильтровал слова. ?

        self.calculate_frequency(s) 
        return s
    
    def calculate_frequency(self, sent_numb: set):
        sent_list = self.get_sentences_by_indexes(sent_numb)
        index = self.create_index(sent_list)
        self.__searched_frequency = self.__topOfIndex(index)
    
    # Дублирование кода с datastorage
    def get_sentences_by_indexes(self, indexes: set) -> list:
        sent_list = []
        for i in indexes:
            sent_list.append(self.__sentences[i])
        return sent_list
    
    def clearState(self):
        """
        Очистить состояние поиска.
        """
        self.__searched_words.clear()
        self.__searched_sentences.clear()
        self.__searched_frequency.clear()

    # TODO: сделать вывод по популяронсти встреч
    def printIndex(self, n = None):
        if n == None or n > len(self.__index):
            print(self.__index)
        else:
            firstOfN = list(self.__index.items())[:n]
            for key, value in firstOfN:
                print(f"{key}: {value}")

    def printResult(self):
        print(f"{self.__searched_words}, {self.__searched_sentences}")
    
    def printWordFrequency(self, n):
        """
        Выводит какие слова и насколько часто встречаются с поисковым(и) словом(ами) из __searched_words
        """
        size = len(self.__searched_frequency)
        if (n > size):
            n = size
        
        for x in self.__searched_frequency[:n]:
            print(x.word, x.freq)


    def __search(self, search_word) -> set:
        if search_word not in self.__index:
            return {}
        else:
            return self.__index[search_word]
        
    def __topOfIndex(self, index: dict) -> list: # TODO: переписать это на dict
        word_frequency = []
        for key in index:
            word_freq = WordFrequency(key, [len(index[key])])
            word_frequency.append(word_freq)
        word_frequency.sort(key=lambda x: x.freq, reverse=True)
        return word_frequency
