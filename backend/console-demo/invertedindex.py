from nltk.tokenize import word_tokenize

class InvertedIndex:
    """
    __index         - словарь вида <string, set<int>>, где ключ - слово, а значение - множество индексов списка предложений (__sentences).\n
    __sentences:    - список предложений.\n

    __searched_words        - слова по которым прошел поиск.\n
    __searched_sentences    - предложения, в которых слова встретились.\n
    """

    __index = dict()
    __sentences: list

    __searched_words = set()
    __searched_sentences = set()

    def __init__(self, sentences: list):
        self.__sentences = sentences
        for i in range(len(sentences)):
            sent = sentences[i]
            words = word_tokenize(sent)
            for word in words:
                if word in self.__index:
                    self.__index[word].add(i)
                else:
                    s = set()
                    s.add(i)
                    self.__index[word] = s
    
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
            return s

        for x in self.__searched_sentences:
            sent = self.__sentences[x]
            words = word_tokenize(sent)
            if search_word in words:
                s.add(x)

        self.__searched_sentences = s.copy()
        self.__searched_words.add(search_word) # Нужно сделать так, чтобы при добавлении set не фильтровал слова.
        return s
    
    def clearState(self):
        """
        Очистить состояние поиска.
        """
        self.__searched_words.clear()
        self.__searched_sentences.clear()

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

    def __search(self, search_word) -> set:
        if search_word not in self.__index:
            return {}
        else:
            return self.__index[search_word]
