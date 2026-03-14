
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

class InvIndex:
    _index: dict[str, set[int]]
    _word_frequency: list[MyWord]
    
    def __init__(self, index: dict[str, set[int]]) -> None:
        self._index = index
        self._word_frequency = self._convertIndexToList(self._index)
    
    def get_word_freq(self, word):
        return len(self._index[word])

    def get_top_word_frequency(self, n = None):
        size = len(self._word_frequency)
        if (n == None or size < n):
            n = size
        return self._word_frequency[:n]

    def get_least_frequent_words(self, n: int | None = None) -> list[MyWord]:
        """
        Вернуть N наименее частых слов.
        """
        if not self._word_frequency:
            return []
        
        if n is None:
            return self._word_frequency[::-1]
        
        return self._word_frequency[-n:][::-1]

    def get_words_by_frequency(self, freq: int) -> list[str]:
        """
        Вернуть все слова с указанной частотой.
        """
        return [w.word for w in self._word_frequency if w.freq == freq]

    def _convertIndexToList(self, index: dict) -> list:
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