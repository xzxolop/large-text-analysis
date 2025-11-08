import math
from collections import Counter
import re

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.documents = {}
        self.original_documents = {}
        self.doc_count = 0
    
    def add_documents(self, documents_df):
        for doc_id, row in documents_df.iterrows():
            self.add_document(doc_id, row['processed'], row['original'])
    
    def add_document(self, doc_id, processed_text, original_text):
        self.documents[doc_id] = processed_text
        self.original_documents[doc_id] = original_text
        self.doc_count += 1
        
        words = self._tokenize(processed_text)
        word_freq = Counter(words)
        total_words = len(words)
        
        for word, freq in word_freq.items():
            tf = freq / total_words
            if word not in self.index:
                self.index[word] = {}
            self.index[word][doc_id] = tf
    
    def get_co_occurring_words_multiple(self, search_words, limit=10):
        """Находит слова, которые встречаются вместе с несколькими словами"""
        if not search_words:
            return []
        
        # Находим документы, где встречаются ВСЕ слова из search_words
        common_docs = None
        for word in search_words:
            if word not in self.index:
                return []  # Если хоть одно слово не найдено, возвращаем пустой список
            
            word_docs = set(self.index[word].keys())
            if common_docs is None:
                common_docs = word_docs
            else:
                common_docs = common_docs.intersection(word_docs)
        
        if not common_docs:
            return []
        
        # Собираем статистику совместного появления с другими словами
        word_freq = {}
        
        for word, doc_weights in self.index.items():
            if word in search_words:
                continue  # Пропускаем сами слова запроса
                
            # Находим документы, где встречаются все слова запроса И текущее слово
            word_common_docs = common_docs.intersection(doc_weights.keys())
            frequency = len(word_common_docs)
            
            if frequency > 0:
                word_freq[word] = frequency
        
        # Сортируем по убыванию частоты
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_words[:limit] if limit > 0 else sorted_words
    
    def search_sentences_with_multiple_words(self, search_words):
        """Находит предложения, содержащие несколько слов"""
        if not search_words:
            return []
        
        # Находим документы, где встречаются ВСЕ слова
        common_docs = None
        for word in search_words:
            if word not in self.index:
                return []
            
            word_docs = set(self.index[word].keys())
            if common_docs is None:
                common_docs = word_docs
            else:
                common_docs = common_docs.intersection(word_docs)
        
        if not common_docs:
            return []
        
        result_sentences = []
        for doc_id in common_docs:
            original_sentence = self.original_documents[doc_id]
            if original_sentence not in [s['original'] for s in result_sentences]:
                result_sentences.append({
                    'original': original_sentence,
                    'processed': self.documents[doc_id]
                })
            
            if len(result_sentences) >= 100:
                break
        
        return result_sentences
    
    def _tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())