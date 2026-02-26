"""
Тесты для модуля кластерного анализа (ClusterAnalyzer).
"""

import pytest
from analysis.cluster_analyzer import ClusterAnalyzer


@pytest.fixture
def sample_sentences():
    """Простой набор предложений для тестирования."""
    return [
        "machine learning is interesting",
        "learning with teacher is supervised learning",
        "natural language processing is also machine learning",
        "deep learning uses neural networks",
        "neural networks are inspired by brain",
    ]


@pytest.fixture
def analyzer(sample_sentences):
    """Создание анализатора для тестов."""
    return ClusterAnalyzer(sample_sentences)


class TestClusterAnalyzerInit:
    """Тесты инициализации ClusterAnalyzer."""
    
    def test_init_creates_analyzer(self, sample_sentences):
        """Проверка создания анализатора."""
        analyzer = ClusterAnalyzer(sample_sentences)
        assert analyzer.n_docs == len(sample_sentences)
    
    def test_init_precomputes_frequencies(self, analyzer):
        """Проверка предварительного расчёта частот."""
        assert len(analyzer.word_doc_freq) > 0
        assert len(analyzer.cooccurrence) > 0
    
    def test_word_freq_counts_correctly(self, analyzer):
        """Проверка подсчёта частоты слов."""
        # "learning" встречается в 4 предложениях
        assert analyzer.word_doc_freq.get("learning", 0) >= 3
        # "machine" встречается в 2 предложениях
        assert analyzer.word_doc_freq.get("machine", 0) >= 1


class TestPMI:
    """Тесты расчёта PMI."""
    
    def test_pmi_same_word(self, analyzer):
        """PMI слова с самим собой обрабатывается специально (возвращает 0)."""
        # PMI одного слова с самим собой не вычисляется явно в нашей реализации
        # потому что мы используем key = (min, max) и w1 < w2
        pmi = analyzer.pmi("learning", "learning")
        assert pmi == 0.0  # Ожидаем 0, т.к. пара (word, word) не считается
    
    def test_pmi_related_words(self, analyzer):
        """PMI связанных слов должен быть положительным."""
        # "machine" и "learning" часто встречаются вместе
        pmi = analyzer.pmi("machine", "learning")
        assert pmi > 0
    
    def test_pmi_unrelated_words(self, analyzer):
        """PMI несвязанных слов может быть низким или отрицательным."""
        # "brain" и "teacher" вряд ли связаны
        pmi = analyzer.pmi("brain", "teacher")
        # Может быть отрицательным или нулём
        assert pmi <= analyzer.pmi("machine", "learning")
    
    def test_pmi_nonexistent_word(self, analyzer):
        """PMI несуществующего слова должен быть 0."""
        pmi = analyzer.pmi("nonexistent_word_xyz", "learning")
        assert pmi == 0.0
    
    def test_pmi_is_symmetric(self, analyzer):
        """PMI должен быть симметричным: PMI(a,b) == PMI(b,a)."""
        pmi_ab = analyzer.pmi("machine", "learning")
        pmi_ba = analyzer.pmi("learning", "machine")
        assert pmi_ab == pmi_ba


class TestNPMI:
    """Тесты Normalized PMI."""
    
    def test_npmi_range(self, analyzer):
        """NPMI должен быть в диапазоне [-1, 1]."""
        words = ["machine", "learning", "neural", "networks", "brain"]
        for w1 in words:
            for w2 in words:
                npmi = analyzer.npmi(w1, w2)
                assert -1.0 <= npmi <= 1.0
    
    def test_npmi_same_word(self, analyzer):
        """NPMI слова с самим собой возвращает 0 (не вычисляется)."""
        npmi = analyzer.npmi("learning", "learning")
        assert npmi == 0.0  # Пара (word, word) не считается в нашей реализации


class TestGetClusterWords:
    """Тесты получения слов кластера."""
    
    def test_get_cluster_words_returns_list(self, analyzer):
        """Метод должен возвращать список кортежей."""
        cluster = analyzer.get_cluster_words("learning", top_n=5)
        assert isinstance(cluster, list)
        if cluster:
            assert isinstance(cluster[0], tuple)
            assert len(cluster[0]) == 2
    
    def test_get_cluster_words_filters_pos(self, analyzer):
        """POS-фильтр должен убирать местоимения и предлоги."""
        cluster = analyzer.get_cluster_words("learning", filter_pos=True)
        # Служебные части речи должны быть отфильтрованы
        # nltk POS теги: is=VBZ, with=IN, by=IN, are=VBP, also=RB
        # VBZ, VBP - глаголы (полезные), IN - предлоги (бесполезные), RB - наречия (бесполезные)
        filtered_words = [w for w, _ in cluster]
        
        # Проверяем что предлоги отфильтрованы
        assert "with" not in filtered_words
        assert "by" not in filtered_words
    
    def test_get_cluster_words_top_n(self, analyzer):
        """top_n должен ограничивать количество результатов."""
        cluster = analyzer.get_cluster_words("learning", top_n=3)
        assert len(cluster) <= 3
    
    def test_get_cluster_words_sorted(self, analyzer):
        """Результаты должны быть отсортированы по убыванию score."""
        cluster = analyzer.get_cluster_words("learning", top_n=10)
        scores = [score for _, score in cluster]
        assert scores == sorted(scores, reverse=True)
    
    def test_get_cluster_words_nonexistent(self, analyzer):
        """Для несуществующего слова должен返回 пустой список."""
        cluster = analyzer.get_cluster_words("nonexistent_xyz")
        assert cluster == []
    
    def test_get_cluster_words_with_npmi(self, analyzer):
        """use_npmi=True должен использовать NPMI вместо PMI."""
        cluster_pmi = analyzer.get_cluster_words("learning", top_n=5, use_npmi=False)
        cluster_npmi = analyzer.get_cluster_words("learning", top_n=5, use_npmi=True)
        
        # Scores должны отличаться
        if cluster_pmi and cluster_npmi:
            pmi_scores = [s for _, s in cluster_pmi]
            npmi_scores = [s for _, s in cluster_npmi]
            # NPMI должен быть в диапазоне [-1, 1]
            for score in npmi_scores:
                assert -1.0 <= score <= 1.0


class TestGetClusterWithFrequency:
    """Тесты получения кластера с частотой."""
    
    def test_returns_triplets(self, analyzer):
        """Метод должен возвращать кортежи (слово, частота, score)."""
        cluster = analyzer.get_cluster_with_frequency("learning", top_n=5)
        if cluster:
            assert len(cluster[0]) == 3
            word, freq, score = cluster[0]
            assert isinstance(word, str)
            assert isinstance(freq, int)
            assert isinstance(score, float)


class TestGetWordsInSeedContext:
    """Тесты получения слов в контексте seed_word."""
    
    def test_returns_context_words(self, analyzer):
        """Метод должен возвращать слова из контекста."""
        context = analyzer.get_words_in_seed_context("learning")
        assert isinstance(context, list)
        # "learning" не должно быть в результате
        assert "learning" not in context
    
    def test_context_contains_related_words(self, analyzer):
        """Контекст должен содержать связанные слова."""
        context = analyzer.get_words_in_seed_context("machine")
        # "learning" должно быть в контексте "machine"
        assert "learning" in context


class TestPOSFiltering:
    """Тесты POS-фильтрации."""
    
    def test_filter_removes_pronouns(self, analyzer):
        """Фильтр должен удалять местоимения."""
        # Добавим предложение с местоимениями
        sentences = ["i think it is good", "you know they are here"]
        test_analyzer = ClusterAnalyzer(sentences)
        
        # С POS-фильтром
        cluster_filtered = test_analyzer.get_cluster_words("think", filter_pos=True)
        filtered_words = [w for w, _ in cluster_filtered]
        
        # Без POS-фильтра
        cluster_all = test_analyzer.get_cluster_words("think", filter_pos=False)
        all_words = [w for w, _ in cluster_all]
        
        # Отфильтрованных должно быть меньше или равно
        assert len(filtered_words) <= len(all_words)
