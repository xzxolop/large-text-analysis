import math
from collections import Counter
import re

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.documents = {}
        self.doc_count = 0
    
    def add_documents(self, documents):
        """Добавляет несколько документов в индекс"""
        for doc_id, text in enumerate(documents):
            self.add_document(doc_id, text)
    
    def add_document(self, doc_id, text):
        """Добавление документа с расчетом TF"""
        self.documents[doc_id] = text
        self.doc_count += 1
        
        words = self._tokenize(text)
        word_freq = Counter(words)
        total_words = len(words)
        
        for word, freq in word_freq.items():
            tf = freq / total_words
            
            if word not in self.index:
                self.index[word] = {}
            
            self.index[word][doc_id] = tf
    
    def get_co_occurring_words(self, search_word, limit=10):
        """Находит слова, которые чаще всего встречаются вместе с search_word"""
        if search_word not in self.index:
            return []
        
        # Собираем статистику совместного появления
        word_freq = {}
        search_word_docs = set(self.index[search_word].keys())
        
        for word, doc_weights in self.index.items():
            if word == search_word:
                continue
                
            # Находим документы, где оба слова встречаются вместе
            common_docs = search_word_docs.intersection(doc_weights.keys())
            frequency = len(common_docs)
            
            if frequency > 0:
                word_freq[word] = frequency
        
        # Сортируем по убыванию частоты
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_words[:limit] if limit > 0 else sorted_words
    
    def search_sentences_with_words(self, search_word, co_occurring_words):
        """Находит предложения, содержащие search_word и co_occurring_words"""
        if search_word not in self.index:
            return []
        
        search_word_docs = set(self.index[search_word].keys())
        result_sentences = []
        
        for word in co_occurring_words:
            if word in self.index:
                # Документы, где встречаются оба слова
                common_docs = search_word_docs.intersection(self.index[word].keys())
                
                for doc_id in common_docs:
                    sentence = self.documents[doc_id]
                    # Добавляем только уникальные предложения
                    if sentence not in result_sentences:
                        result_sentences.append(sentence)
                    
                    # Ограничиваем количество результатов для производительности
                    if len(result_sentences) >= 100:  # максимум 100 предложений
                        return result_sentences
        
        return result_sentences
    
    def _tokenize(self, text):
        """Токенизация текста"""
        return re.findall(r'\b\w+\b', text.lower())
