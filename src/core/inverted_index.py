from nltk.tokenize import word_tokenize

class MyWord:
    word: str
    freq: int

    def __init__(self, word, freq):
        self.word = word
        self.freq = freq

    def __eq__(self, other):  # == (equal)
        return self.word == other.word and self.freq == other.freq

    def __str__(self):
        return str(f"{self.word}: {self.freq}")

    def __repr__(self):
        # Чтобы при печати списков/словарей (используют repr) отображалось человекочитаемо
        return f"{self.word}: {self.freq}"

class SearchState:
    """
    Состояние поиска: хранит слова, предложения и связанные слова.

    Attributes:
        searched_words: Слова, по которым был выполнен поиск.
        searched_sentences: Индексы предложений, где найдены слова.
        word_frequency: Список связанных слов, отсортированный по частоте.
    """

    searched_words: set[str]
    searched_sentences: set[int]
    word_frequency: list[MyWord]

    def __init__(self) -> None:
        self.searched_words = set()
        self.searched_sentences = set()
        self.word_frequency = []

    def clear_state(self):
        """
        Очистить состояние поиска.
        """
        self.searched_words.clear()
        self.searched_sentences.clear()
        self.word_frequency.clear()

    def print_word_frequency(self, n = None):
        """
        Выводит какие слова и насколько часто встречаются с поисковым(и) словом(ами)
        """
        size = len(self.word_frequency)
        if (n == None or size < n):
            n = size

        print(self.word_frequency[:n])

    def print_matches(self):
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

    __index: dict[str, set[int]]
    __sentences: list[str]
    __word_frequency: list[MyWord]  # NOTE: нужен чтобы каждый раз не пересчитывать.

    def __init__(self, sentences: list, calc_word_freq: bool = False):
        self.__sentences = sentences
        self.__index = self.__create_index(sentences)
        self.__word_frequency = []

        if calc_word_freq:
            self.__word_frequency = self.__convertIndexToList(self.__index)

    def __create_index(self, sentences: list) -> dict:
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

    def get_top_word_frequency(self, n = None):
        size = len(self.__word_frequency)
        if (n == None or size < n):
            n = size

        return self.__word_frequency[:n]

    def get_least_frequent_words(self, n: int | None = None) -> list[MyWord]:
        """
        Вернуть N наименее частых слов.
        
        Args:
            n: Количество слов. Если None — вернуть все слова в порядке возрастания частоты.
        """
        if not self.__word_frequency:
            return []
        
        if n is None:
            return self.__word_frequency[::-1]
        
        return self.__word_frequency[-n:][::-1]

    def get_words_by_frequency(self, freq: int) -> list[str]:
        """
        Вернуть все слова с указанной частотой.
        
        Args:
            freq: Частота слова (количество вхождений).
        
        Returns:
            Список слов с заданной частотой.
        """
        return [w.word for w in self.__word_frequency if w.freq == freq]

    # NOTE: как будто не сильно нужно. Эту ответственность можно возлажить на пользователя
    def print_top_word_frequency(self, n = None):
        """
        Выводит наиболее популярные слова среди загруженных предложений.
        """
        lst = self.get_top_word_frequency(n)

        for x in lst:
            print(x)

    # NOTE: как будто не сильно нужен. Можно оставить для дебага.
    def print_index(self, n = None):
        """
        Печатает весь индекс как структуру данных.
        """
        if n == None or n > len(self.__index):
            print(self.__index)
        else:
            firstOfN = list(self.__index.items())[:n]
            for key, value in firstOfN:
                print(f"{key}: {value}")

    def __calculate_frequency(self, indexes: set) -> list:
        """
        Эта функция нужна, чтобы для поискового слова (например python) найти слова которые чаще всего с ним встречаются.
        На вход принимает номера предожений в которых встречается поисковое слово.
        """
        sent_list = self.__get_sentences_by_indexes(indexes)
        index = self.__create_index(sent_list)
        return self.__convertIndexToList(index)

    def __search(self, search_word) -> set:
        if search_word not in self.__index:
            return {}
        else:
            return self.__index[search_word]

    def __get_sentences_by_indexes(self, indexes: set) -> list:
        """
        Возвращает список предложений по набору индексов.
        """
        return [self.__sentences[i] for i in indexes]

    def __convertIndexToList(self, index: dict) -> list:
        """
        Преобразует инвертированный индекс в список, отсоритрованный по популярности встреч слова в предложениях.
        """
        wordsList = []
        for word in index:
            word_freq = len(index[word])

            if word_freq == 0:
                continue

            myWord = MyWord(word, word_freq)
            wordsList.append(myWord)
        wordsList.sort(key=lambda x: x.freq, reverse=True)
        return wordsList
