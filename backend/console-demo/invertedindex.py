from nltk.tokenize import word_tokenize

class InvertedIndex:
    index = dict()

    def __init__(self, sentences: list):
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
        
    
    def search(self, search_word):
        if search_word not in self.index:
            return {}
        else:
            return self.index[search_word]
    
    def print(self):
        print(self.index)
    
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

w = idx.search("Кот")
print("Кот:", w)