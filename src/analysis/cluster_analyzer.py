"""
Модуль для кластерного анализа слов на основе PMI (Pointwise Mutual Information).

PMI измеряет силу ассоциации между словами:
  PMI(w1, w2) = log(P(w1, w2) / (P(w1) * P(w2)))

Высокий PMI означает, что слова встречаются вместе чаще, чем случайно.
"""

from collections import Counter
from typing import List, Tuple, Optional
import math
import nltk
from nltk import pos_tag, word_tokenize


class ClusterAnalyzer:
    """
    Анализ кластеров слов на основе PMI и POS-фильтрации.
    
    Пример использования:
        analyzer = ClusterAnalyzer(sentences)
        cluster = analyzer.get_cluster_words("russia", top_n=20)
        # [('war', 2.5), ('country', 2.1), ('world', 1.8), ...]
    """
    
    # POS-теги для информативных слов (существительные, глаголы, прилагательные)
    USEFUL_POS = {
        'NN', 'NNS', 'NNP', 'NNPS',    # существительные
        'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # глаголы
        'JJ', 'JJR', 'JJS',  # прилагательные
    }
    
    def __init__(self, sentences: List[str]):
        """
        Инициализация анализатора.
        
        Args:
            sentences: Список предложений для анализа.
        """
        self.sentences = sentences
        self.n_docs = len(sentences)
        
        # Частота слов (в скольких документах встречается, все равно что сумма значений для ключа в inverted index)
        self.word_doc_freq: Counter = Counter()

        # Общая частота слов (для нормализации)
        self.word_total_freq: Counter = Counter()
        
        # Частота совместной встречаемости пар слов
        self.cooccurrence: Counter = Counter()
        
        self._precompute()
    
    def _precompute(self):
        """
        Предварительный расчёт частот для ускорения PMI.
        
        Вычисляет:
        - word_doc_freq: в скольких документах встречается каждое слово
        - cooccurrence: сколько раз пара слов встречается в одном документе
        - word_total_freq: общее количество вхождений каждого слова
        """
        for sent in self.sentences:
            words = word_tokenize(sent.lower())
            
            # Учитываем только уникальные слова в предложении для doc_freq
            unique_words = set(words)
            for w in unique_words:
                self.word_doc_freq[w] += 1
            
            # Общее количество вхождений
            for w in words:
                self.word_total_freq[w] += 1
            
            # Совместная встречаемость пар
            for w1 in unique_words:
                for w2 in unique_words:
                    if w1 < w2:  # чтобы не дублировать (w1,w2) и (w2,w1)
                        self.cooccurrence[(w1, w2)] += 1
    
    def pmi(self, word1: str, word2: str) -> float:
        """
        Вычислить PMI между двумя словами.
        
        Args:
            word1: Первое слово.
            word2: Второе слово.
            
        Returns:
            Значение PMI (логарифм отношения совместной вероятности к произведению маргинальных).
        """
        word1 = word1.lower()
        word2 = word2.lower()
        
        # Вероятности P(word)
        p1 = self.word_doc_freq.get(word1, 0) / self.n_docs
        p2 = self.word_doc_freq.get(word2, 0) / self.n_docs
        
        # Совместная вероятность P(word1, word2)
        key = (min(word1, word2), max(word1, word2)) # min, max нужны в таком порядке т.к. мы сравнивали w1 < w2 в методе _precompute
        cooccur = self.cooccurrence.get(key, 0)
        p_joint = cooccur / self.n_docs
        
        if p_joint == 0 or p1 == 0 or p2 == 0:
            return 0.0
        
        return math.log(p_joint / (p1 * p2))
    
    def npmi(self, word1: str, word2: str) -> float:
        """
        Вычислить Normalized PMI (NPMI) между двумя словами.
        
        NPMI нормализует PMI к диапазону [-1, 1]:
          NPMI(w1, w2) = PMI(w1, w2) / -log(P(w1, w2))
        
        Args:
            word1: Первое слово.
            word2: Второе слово.
            
        Returns:
            Значение NPMI в диапазоне [-1, 1].
        """
        word1 = word1.lower()
        word2 = word2.lower()
        
        key = (min(word1, word2), max(word1, word2))
        cooccur = self.cooccurrence.get(key, 0)
        
        if cooccur == 0:
            return 0.0
        
        p_joint = cooccur / self.n_docs
        pmi = self.pmi(word1, word2)
        
        # Нормализация
        nPMI = pmi / (-math.log(p_joint)) if p_joint > 0 else 0.0
        
        return nPMI
    
    def _filter_by_pos(self, words: List[str]) -> List[str]:
        """
        Отфильтровать слова по частям речи.
        
        Args:
            words: Список слов для фильтрации.
            
        Returns:
            Список слов с полезными POS-тегами.
        """
        tagged = pos_tag(words)
        return [word for word, pos in tagged if pos in self.USEFUL_POS]
    
    def get_cluster_words(
        self,
        seed_word: str,
        top_n: int = 20,
        min_pmi: float = 0.0,
        filter_pos: bool = True,
        use_npmi: bool = False,
        min_freq: int = 1,
        tfidf_range: Optional[Tuple[float, float]] = None,
        word_tfidf_scores: Optional[dict] = None,
        use_freq_weighting: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Получить слова кластера для заданного слова.
        
        Args:
            seed_word: Исходное слово (запрос пользователя).
            top_n: Количество возвращаемых результатов.
            min_pmi: Минимальный PMI для фильтрации.
            filter_pos: Фильтровать по частям речи (существительные, глаголы, прилагательные).
            use_npmi: Использовать Normalized PMI вместо обычного.
            min_freq: Минимальная частота слова (в скольких документах встречается).
            tfidf_range: Диапазон TF-IDF (min, max) для фильтрации. 
                         Например, (0.1, 0.85) отсечет нерелевантные слова.
            word_tfidf_scores: Словарь {слово: tfidf_score} для TF-IDF фильтрации.
            use_freq_weighting: Если True, использовать комбинированный скор PMI × log(freq).
            
        Returns:
            Список кортежей (слово, score), отсортированный по убыванию score.
        """
        seed_word = seed_word.lower()
        
        # Проверяем, есть ли слово в корпусе
        if seed_word not in self.word_doc_freq:
            return []
        
        scores = []
        
        for word in self.word_doc_freq:
            if word == seed_word:
                continue
            
            # Фильтр по минимальной частоте (мягкий, по умолчанию = 1)
            if self.word_doc_freq[word] < min_freq:
                continue
            
            # POS-фильтр
            if filter_pos:
                tagged = pos_tag([word])
                if tagged[0][1] not in self.USEFUL_POS:
                    continue
            
            # TF-IDF фильтр
            if tfidf_range is not None and word_tfidf_scores is not None:
                tfidf = word_tfidf_scores.get(word, 0.0)
                tfidf_min, tfidf_max = tfidf_range
                if tfidf < tfidf_min or tfidf > tfidf_max:
                    continue
            
            # Вычисляем PMI
            if use_npmi:
                score = self.npmi(seed_word, word)
            else:
                score = self.pmi(seed_word, word)
            
            # Комбинированный скор: PMI × log(freq)
            if use_freq_weighting:
                freq = self.word_doc_freq[word]
                score = score * math.log(freq + 1)
            
            if score >= min_pmi:
                scores.append((word, score))
        
        # Сортировка по убыванию
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_n]
    
    def get_cluster_with_frequency(
        self,
        seed_word: str,
        top_n: int = 20,
        min_pmi: float = 0.0,
        filter_pos: bool = True,
    ) -> List[Tuple[str, int, float]]:
        """
        Получить слова кластера с частотой встречаемости.
        
        Args:
            seed_word: Исходное слово.
            top_n: Количество результатов.
            min_pmi: Минимальный PMI.
            filter_pos: Фильтровать по POS.
            
        Returns:
            Список кортежей (слово, частота, pmi_score).
        """
        cluster = self.get_cluster_words(seed_word, top_n * 2, min_pmi, filter_pos)
        
        result = []
        for word, pmi_score in cluster:
            freq = self.word_doc_freq.get(word, 0)
            result.append((word, freq, pmi_score))
        
        # Сортируем по PMI, но возвращаем с частотой
        result.sort(key=lambda x: x[2], reverse=True)
        
        return result[:top_n]
    
    def get_words_in_seed_context(self, seed_word: str) -> List[str]:
        """
        Получить все уникальные слова, которые встречаются в предложениях с seed_word.
        
        Args:
            seed_word: Исходное слово.
            
        Returns:
            Список уникальных слов из контекста seed_word.
        """
        seed_word = seed_word.lower()
        context_words = set()
        
        for sent in self.sentences:
            sent_lower = sent.lower()
            if seed_word in sent_lower.split():
                words = word_tokenize(sent_lower)
                context_words.update(words)
        
        context_words.discard(seed_word)
        return list(context_words)
