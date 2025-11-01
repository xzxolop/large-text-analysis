import math
from collections import Counter
import re

class AdvancedInvertedIndex:
    def __init__(self):
        self.index = {}  # слово -> {doc_id: tf}
        self.documents = {}  # doc_id -> текст
        self.doc_count = 0
        self.co_occurrence_cache = {}  # Кэш для часто встречающихся слов
    
    def add_documents_serias(self, documents_series):
        """Добавление документов из Series с авто-генерацией ID"""
        for doc_id, text in documents_series.items():
            self.add_document(doc_id, text)
    
    def add_documents(self, documents):
        for doc_id, text in documents.items():
            self.add_document(doc_id, text)
    
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
    
    def search_documents(self, query):
        """Поиск документов, содержащих все слова запроса"""
        words = self._tokenize(query)
        
        if not words:
            return set()
        
        # Начинаем с первого слова
        result = set(self.index.get(words[0], {}).keys())
        
        # Пересечение множеств для всех слов (AND логика)
        for word in words[1:]:
            result = result.intersection(self.index.get(word, {}).keys())
        
        return result
    
    def find_co_occurring_words(self, target_word, top_k=10):
        """Найти слова, которые чаще всего встречаются вместе с target_word"""
        if target_word not in self.index:
            return []
        
        # Документы, содержащие целевое слово
        target_docs = set(self.index[target_word].keys())
        
        # Счетчик для слов, встречающихся в тех же документах
        co_occurrence_counter = Counter()
        
        for word, doc_tf in self.index.items():
            if word == target_word:
                continue
                
            # Документы, содержащие текущее слово
            word_docs = set(doc_tf.keys())
            
            # Количество документов, где оба слова встречаются вместе
            co_occurrence_count = len(target_docs.intersection(word_docs))
            
            if co_occurrence_count > 0:
                # Используем TF-IDF как меру важности
                idf = math.log(self.doc_count / (1 + len(word_docs)))
                avg_tf = sum(doc_tf.values()) / len(doc_tf)
                score = co_occurrence_count * avg_tf * idf
                
                co_occurrence_counter[word] = score
        
        return co_occurrence_counter.most_common(top_k)
    
    def get_documents_with_word(self, word):
        """Получить документы, содержащие слово"""
        if word not in self.index:
            return []
        
        doc_ids = list(self.index[word].keys())
        documents = []
        
        for doc_id in doc_ids:
            documents.append({
                'doc_id': doc_id,
                'text': self.documents.get(doc_id, ''),
                'tf': self.index[word][doc_id]
            })
        
        return documents
    
    def _tokenize(self, text):
        """Улучшенная токенизация"""
        if not isinstance(text, str):
            return []
        # Удаление знаков препинания и приведение к нижнему регистру
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def get_word_statistics(self, word):
        """Получить статистику по слову"""
        if word not in self.index:
            return None
        
        doc_freq = len(self.index[word])
        idf = math.log(self.doc_count / (1 + doc_freq))
        avg_tf = sum(self.index[word].values()) / doc_freq
        
        return {
            'document_frequency': doc_freq,
            'idf': idf,
            'average_tf': avg_tf,
            'total_occurrences': sum(len(self._tokenize(self.documents[doc_id])) for doc_id in self.index[word])
        }