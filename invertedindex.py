class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.documents = {}

    def add_documents(self, documents):
        for doc_id, text in documents.items():
            self.add_document(doc_id, text)
    
    def add_document(self, doc_id, text):
        """Добавление документа в индекс"""
        self.documents[doc_id] = text
        
        # Токенизация и нормализация текста
        words = self._tokenize(text)
        
        for word in words:
            if word not in self.index:
                self.index[word] = set()
            self.index[word].add(doc_id)
    
    def search(self, query):
        """Поиск документов по запросу"""
        words = self._tokenize(query)
        
        if not words:
            return set()
        
        # Начинаем с первого слова
        result = self.index.get(words[0], set())
        
        # Пересечение множеств для всех слов (AND логика)
        for word in words[1:]:
            result = result.intersection(self.index.get(word, set()))
        
        return result
    
    def _tokenize(self, text):
        """Простая токенизация текста"""
        # Приведение к нижнему регистру и разбиение по пробелам
        return text.lower().split()
    
    def get_document_frequency(self, word):
        """Количество документов, содержащих слово"""
        return len(self.index.get(word, set()))
    
    def print_index(self):
        """Вывод индекса на экран"""
        for word, doc_ids in sorted(self.index.items()):
            print(f"{word}: {sorted(doc_ids)}")