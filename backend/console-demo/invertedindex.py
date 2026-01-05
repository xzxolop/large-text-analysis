from nltk.tokenize import word_tokenize

class InvertedIndex:
    index = dict()
    _sentences: list

    searched_words = set()
    searched_sentences = set()

    def __init__(self, sentences: list):
        self._sentences = sentences
        for i in range(len(sentences)):
            sent = sentences[i]
            words = word_tokenize(sent)
            for word in words:
                if word in self.index:
                    self.index[word].add(i)
                else:
                    s = set()
                    s.add(i)
                    self.index[word] = s
        
    def search(self, search_word) -> set:
        if search_word not in self.index:
            return {}
        else:
            return self.index[search_word]
        
    def searchWith(self, search_word) -> set:
        if search_word in self.searched_words:
            return {} # TODO: мб кинуть ошибку
        
        s = set()
        if len(self.searched_words) == 0:
            s = self.search(search_word) # NOTE: это плохо. Возможна потеря данных!
            self.searched_words.add(search_word)
            self.searched_sentences = s.copy()
            return s

        for x in self.searched_sentences:
            sent = self._sentences[x]
            words = word_tokenize(sent)
            if search_word in words:
                s.add(x)

        self.searched_sentences = s.copy()
        self.searched_words.add(search_word)
        return s
    
    def print(self):
        print(self.index)

    def printResult(self):
        print(self.searched_words, self.searched_sentences)

    def getSentByIndex(self, index: int) -> list:
        return self._sentences[index].copy()

    def getSentByIndexes(self, indexes: set) -> list:
        sent_list = []
        for i in indexes:
            sent_list.append(self._sentences[i].copy())
        return sent_list
    
sentences = [
    "Кот сидит на ковре.",
    "Собака бежит за котом.",
    "Кот и собака играют.",
    "Рыжий кот спит на диване.",
    "Собака охраняет дом."
]

# Создаем инвертированный индекс
idx = InvertedIndex(sentences)
idx.print()

#w = idx.search("Кот")
#print("Кот:", w)

w = idx.searchWith("Кот")
print("w:", w)
idx.printResult()

w = idx.searchWith("сидит")
print("w:", w)
idx.printResult()