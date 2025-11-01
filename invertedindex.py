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



import math
from collections import Counter

class AdvancedInvertedIndex:
    def __init__(self):
        self.index = {}
        self.documents = {}
        self.doc_count = 0
    
    def add_document(self, doc_id, text):
        """Добавление документа с расчетом TF"""
        self.documents[doc_id] = text
        self.doc_count += 1
        
        words = self._tokenize(text)
        word_freq = Counter(words)
        total_words = len(words)
        
        for word, freq in word_freq.items():
            tf = freq / total_words  # Term Frequency
            
            if word not in self.index:
                self.index[word] = {}
            
            self.index[word][doc_id] = tf
    
    def search(self, query, k=10):
        """Поиск с релевантностью по TF-IDF"""
        words = self._tokenize(query)
        
        scores = {}
        
        for word in words:
            if word not in self.index:
                continue
            
            # IDF calculation
            doc_freq = len(self.index[word])
            idf = math.log(self.doc_count / (1 + doc_freq))
            
            for doc_id, tf in self.index[word].items():
                tf_idf = tf * idf
                
                if doc_id not in scores:
                    scores[doc_id] = 0
                scores[doc_id] += tf_idf
        
        # Сортировка по релевантности
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:k]
    
    def _tokenize(self, text):
        """Улучшенная токенизация"""
        import re
        # Удаление знаков препинания и приведение к нижнему регистру
        words = re.findall(r'\b\w+\b', text.lower())
        return words