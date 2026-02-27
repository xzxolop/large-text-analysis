# Кластерный анализ слов: Путь разработки

## 🎯 Постановка задачи

**Исходная проблема:** При поиске слова (например, `russia`) программа возвращала просто слова, отсортированные по частоте совместных вхождений:

```
russia → [russia: 23, china: 4, us: 3, world: 3, ...]
```

**Цель:** Находить не просто частые слова, а **слова, задающие семантический кластер** — тематически связанные понятия.

> Пример: для `дерево` кластер — `берёза`, `дуб`, `листва`, `кора` (гипонимы и связанные понятия)

---

## 📊 Этап 1: Анализ проблемы

### Первая идея: KMeans кластеризация документов

Изначально рассмотрели использование `KMeans` для кластеризации документов по TF-IDF:

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)
kmeans.fit(tfidf_matrix)
```

**Проблема:** KMeans группирует **документы**, а не **слова**. Нам нужно было найти слова, ассоциированные с запросом, а не похожие предложения.

---

## 🔍 Этап 2: Выбор метрики ассоциации

### Решение: PMI (Pointwise Mutual Information)

**Формула:**
$$\text{PMI}(w_1, w_2) = \log \frac{P(w_1, w_2)}{P(w_1) \cdot P(w_2)}$$

**Интерпретация:**
- `PMI > 0` — слова встречаются вместе чаще, чем случайно
- `PMI = 0` — независимы
- `PMI < 0` — реже, чем случайно

**Почему PMI:**
- ✅ Измеряет **силу ассоциации**, а не просто частоту
- ✅ Автоматически penalizes слова, встречающиеся везде
- ✅ Простая интерпретация

---

## 🏗️ Этап 3: Первая реализация

### Создан модуль `src/analysis/cluster_analyzer.py`

**Ключевые компоненты:**

1. **Предварительный расчёт частот:**
   - `word_doc_freq` — в скольких документах встречается каждое слово
   - `cooccurrence` — сколько раз пара слов встречается вместе

2. **Расчёт PMI:**
   ```python
   def pmi(self, word1, word2):
       p1 = self.word_doc_freq[word1] / n_docs
       p2 = self.word_doc_freq[word2] / n_docs
       p_joint = cooccurrence[(word1, word2)] / n_docs
       return log(p_joint / (p1 * p2))
   ```

3. **POS-фильтрация:**
   - Оставляем только существительные (`NN`, `NNS`, `NNP`), глаголы (`VB*`), прилагательные (`JJ*`)
   - Удаляем местоимения, предлоги, союзы

### Первые результаты

```
🔬 Кластер для слова: 'russia'
--------------------------------------------------
   Слово                     Score
--------------------------------------------------
   bitcoins                 8.7173
   seize                    8.7173
   churchill                8.7173
   ww1                      8.7173
   crimea                   8.7173
   nato                     8.0242
```

---

## ⚠️ Этап 4: Обнаружение проблемы

### Проблема №1: PMI любит редкие слова

**Наблюдение:** Слова с одинаковым PMI (8.7173) встречаются **одинаковое число раз** (3-5 раз в корпусе).

**Почему:** Если слово `w2` встречается очень редко (маленькое `P(w2)`), то знаменатель `P(w1) × P(w2)` мал, и PMI **огромен**.

**Пример:**
- `bellingcat` встречается 3 раза, и все 3 — с `russia`
- PMI(`russia`, `bellingcat`) = **8.7** (очень высокий!)
- Но это **не релевантный кластер**, а редкое имя собственное

### Проблема №2: Отсутствие релевантных слов

Хорошие слова (`china`, `country`, `politics`) имели **меньший PMI**, чем редкий мусор, потому что встречаются чаще.

---

## 🔧 Этап 5: Попытки решения

### Попытка №1: TF-IDF фильтр

**Идея:** Отсечь слова с экстремальными TF-IDF (< 0.1 или > 0.85).

**Реализация:**
```python
def get_cluster_words(..., tfidf_range=(0.1, 0.85), word_tfidf_scores=None):
    if tfidf_range and word_tfidf_scores:
        tfidf = word_tfidf_scores.get(word, 0.0)
        if tfidf < 0.1 or tfidf > 0.85:
            continue  # Пропускаем слово
```

**Результат:**
```
🔬 Кластер для слова: 'russia' (TF-IDF фильтр)
--------------------------------------------------
   bellingcat               7.6187  # Всё ещё мусор!
   populist                 7.6187
   china                    7.3310  # ✅ Релевантно
   civilians                7.1079  # ✅ Релевантно
```

**Вывод:** TF-IDF фильтр **не решает проблему** — редкие слова всё ещё проходят фильтр.

---

### Попытка №2: Увеличение `min_freq`

**Идея:** Отсечь слова, встречающиеся реже N раз.

**Проблема:** Для редких слов в датасете (например, `russia` встречается не так часто) высокий `min_freq` **уничтожает весь кластер**.

---

## 💡 Этап 6: Финальное решение

### Комбинированный скор: PMI × log(freq)

**Ключевая инсайт:** Вместо жёсткой фильтрации по частоте — **мягко штрафизировать редкие слова** через логарифм частоты.

**Формула:**
$$\text{score}(w_1, w_2) = \text{PMI}(w_1, w_2) \times \log(\text{freq}(w_2) + 1)$$

**Почему логарифм:**
- ✅ Медленно растёт — не подавляет PMI полностью
- ✅ Поднимает частые слова выше редких
- ✅ Сохраняет редкие, но релевантные слова (не отбрасывает)

**Реализация:**
```python
def get_cluster_words(..., use_freq_weighting=True):
    score = self.pmi(seed_word, word)
    
    if use_freq_weighting:
        freq = self.word_doc_freq[word]
        score = score * math.log(freq + 1)  # Комбинированный скор
```

---

## 🎉 Этап 7: Итоговые результаты

### До (просто PMI):
```
🔬 Кластер для слова: 'russia' (PMI)
--------------------------------------------------
   bellingcat               8.7173
   populist                 8.7173
   novelist                 8.7173
   dbo                      8.7173
   china                    7.3310  # ✅
   civilians                7.1079  # ✅
```

### После (PMI × log(freq)):
```
🔬 Кластер для слова: 'russia' (PMI × log(freq))
--------------------------------------------------
   china                          25.3003  # ✅
   russian                        22.3617  # ✅
   country                        22.1138  # ✅
   world                          21.6075  # ✅
   politics                       21.4550  # ✅
   western                        21.1494  # ✅
   europe                         19.0513  # ✅
```

---

## 📊 Сравнение для всех слов

| Слово | До (PMI) | После (PMI × log(freq)) |
|-------|----------|-------------------------|
| **russia** | bellingcat, dbo, ww1 | ✅ china, russian, country, world, politics |
| **data** | datastack, nans, ipcc | ✅ sets, raw, science, mining, weather |
| **python** | pcpartpicker, astropy | ✅ r, library, pandas, script, module |
| **ai** | neuton, vagueness, voila | ✅ train, training, model, learning, algorithms |

---

## 🏗️ Архитектура решения

```
src/
├── analysis/
│   └── cluster_analyzer.py    # PMI, NPMI, кластеризация
├── search/
│   └── search_engine.py       # Фасад с интеграцией ClusterAnalyzer
└── demo.py                    # show_word_cluster()
```

### Ключевые классы

```python
class ClusterAnalyzer:
    """Анализ кластеров на основе PMI."""
    
    def __init__(self, sentences: List[str])
    def pmi(word1, word2) -> float
    def npmi(word1, word2) -> float  # Normalized PMI [-1, 1]
    def get_cluster_words(
        seed_word,
        top_n=20,
        min_freq=1,
        filter_pos=True,
        use_freq_weighting=True,  # PMI × log(freq)
        tfidf_range=None
    ) -> List[Tuple[str, float]]
```

---

## 🧪 Тестирование

**20 тестов** покрывают:
- ✅ Расчёт PMI (симметрия, связанные/несвязанные слова)
- ✅ NPMI (диапазон [-1, 1])
- ✅ POS-фильтрация
- ✅ Фильтр по частоте (`min_freq`)
- ✅ TF-IDF фильтр
- ✅ Взвешивание по частоте (`use_freq_weighting`)

```bash
uv run pytest tests/test_cluster_analyzer.py -v
# 20 passed
```

---

## 📋 Итоговые параметры по умолчанию

```python
demo.show_word_cluster(
    engine,
    "russia",
    top_n=20,
    min_freq=1,              # Мягкий фильтр (не отбрасывает)
    tfidf_range=None,        # Отключен по умолчанию (медленно)
    use_freq_weighting=True  # ✅ Включено!
)
```

---

## 🔮 Планы на будущее

| Улучшение | Описание |
|-----------|----------|
| **Кэширование PMI** | Для ускорения повторных запросов |
| **Визуализация** | Графы кластеров (networkx + matplotlib) |
| **Иерархия кластеров** | Древовидная структура (дерево → берёза/дуб) |
| **Word2Vec/FastText** | Семантические эмбеддинги извне |
| **Контекстные эмбеддинги** | BERT / Sentence Transformers для учёта контекста |

---

## 📚 Выводы

1. **PMI сам по себе недостаточен** — любит редкие слова
2. **Жёсткие фильтры (min_freq) опасны** — могут удалить релевантные слова
3. **Комбинированный скор (PMI × log(freq))** — оптимальный баланс
4. **POS-фильтрация обязательна** — удаляет служебные части речи
5. **TF-IDF фильтр** — полезен, но медленный на больших корпусах

---

## 🚀 Использование

```python
from data.data_storage import DataStorage
from search.search_engine import SearchEngine
import demo

data_store = DataStorage()
data_store.load_data()
sentences = data_store.get_processed_sentences()

engine = SearchEngine(sentences, enable_cluster_analysis=True)

# Получить кластер для слова
cluster = engine.get_cluster_words("russia", top_n=20)

# Красивый вывод
demo.show_word_cluster(engine, "russia", top_n=20)
```

---

**Дата обновления:** 2026-02-26  
**Авторы:** Разработка в ходе совместной сессии с Qwen Code
