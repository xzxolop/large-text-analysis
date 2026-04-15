"""
Веб-демо для large-text-analysis с использованием FastAPI.

Запуск:
    uv run uvicorn src.web_app:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List, Dict, Set, Tuple
from functools import lru_cache
import json
import sys
from pathlib import Path

# Добавляем src в путь для импортов
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from data.data_storage import DataStorage
from search.search_engine import SearchEngine


# ============================================
# Модели данных для API
# ============================================

class ClusterRequest(BaseModel):
    word: str
    min_freq: Optional[int] = 1
    min_score_percent: Optional[float] = 30.0


class ClusterWord(BaseModel):
    word: str
    freq: int
    score: float


class ClusterStats(BaseModel):
    count: int
    max_score: float
    mean_score: float
    median_score: float
    min_score: float


class ClusterResponse(BaseModel):
    word: str
    cluster: list[ClusterWord]
    stats: ClusterStats


class ExclusiveClusterItem(BaseModel):
    word: str
    freq: int


class ExclusiveClusterResponse(BaseModel):
    clusters: list[ExclusiveClusterItem]


class IterativeRequest(BaseModel):
    seed_words: List[str]


class PmiClusterRequest(BaseModel):
    word: str
    min_freq: Optional[int] = 1
    min_score_percent: Optional[float] = 30.0
    sentence_indices: Optional[List[int]] = None  # Если указано — ищем в подмножестве


class PmiClusterWord(BaseModel):
    word: str
    freq: int
    score: float


class PmiClusterResponse(BaseModel):
    word: str
    cluster: list[PmiClusterWord]
    stats: ClusterStats
    sentence_indices: Optional[List[int]] = None  # Индексы предложений для контекста


class PmiSentencesRequest(BaseModel):
    word: str
    seed_words: List[str] = []
    limit: int = 20
    offset: int = 0


class SentenceItem(BaseModel):
    index: int
    text: str
    highlight_positions: dict[str, list[int]] = {}  # {word: [positions]} для всех слов пути
    search_path: list[str] = []  # Полный путь итеративного поиска


class SentencesResponse(BaseModel):
    word: str
    total_count: int
    sentences: list[SentenceItem]


# ============================================
# Инициализация приложения
# ============================================

app = FastAPI(
    title="Large Text Analysis",
    description="Word Cluster Analysis Demo",
    version="0.1.0",
)

# Настройка шаблонов
templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


# ============================================
# Глобальное состояние (загружается при старте)
# ============================================

class AppState:
    """Хранит инициализированный SearchEngine и данные."""
    engine: Optional[SearchEngine] = None
    data_store: Optional[DataStorage] = None
    all_clusters: Optional[Dict[str, Set[int]]] = None  # Кэш всех кластеров
    cluster_cache: Dict[Tuple[str, ...], Dict[str, Set[int]]] = {}  # LRU кэш: seed_words -> clusters
    cluster_cache_order: List[Tuple[str, ...]] = []  # Порядок использования для LRU
    loaded: bool = False
    MAX_CACHE_SIZE = 20  # Максимальный размер LRU кэша


state = AppState()


@app.on_event("startup")
async def startup_event():
    """Загружает данные и инициализирует SearchEngine при старте приложения."""
    print("🔄 Loading dataset and initializing SearchEngine...")

    state.data_store = DataStorage()
    state.data_store.load_data()  # Загрузка Reddit dataset
    sentences = state.data_store.get_processed_sentences()

    print(f"📄 Loaded {len(sentences)} sentences")

    state.engine = SearchEngine(sentences, calc_word_freq=True)
    state.loaded = True

    print("✅ SearchEngine ready!")


# ============================================
# Endpoints
# ============================================

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Главная страница с формой поиска (PMI clustering)."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/exclusive", response_class=HTMLResponse)
async def read_exclusive(request: Request):
    """Страница exclusive clustering."""
    return templates.TemplateResponse("exclusive.html", {"request": request})


@app.get("/pmi_sequential", response_class=HTMLResponse)
async def read_pmi_sequential(request: Request):
    """Страница PMI sequential clustering."""
    return templates.TemplateResponse("pmi_sequential.html", {"request": request})


@app.post("/api/cluster", response_model=ClusterResponse)
async def get_cluster(request: ClusterRequest):
    """
    Получить кластер слов для заданного слова (PMI-based).
    """
    if not state.loaded or state.engine is None:
        raise HTTPException(status_code=503, detail="Service not ready. Please wait for startup.")

    engine = state.engine
    word = request.word.lower().strip()

    if not word:
        raise HTTPException(status_code=400, detail="Word cannot be empty")

    # Получаем кластер с использованием параметров по умолчанию
    cluster = engine.get_cluster_words(
        seed_word=word,
        top_n=20 * 5,
        use_npmi=False,
        min_freq=request.min_freq or 1,
        tfidf_range=None,
        use_freq_weighting=True,
        min_score_percent=request.min_score_percent or 30.0,
    )

    if not cluster:
        return ClusterResponse(
            word=word,
            cluster=[],
            stats=ClusterStats(
                count=0,
                max_score=0.0,
                mean_score=0.0,
                median_score=0.0,
                min_score=0.0,
            ),
        )

    # Добавляем частоту к каждому слову
    cluster_with_freq = []
    for cluster_word, score in cluster:
        freq = engine._cluster_analyzer.word_doc_freq.get(cluster_word.lower(), 0)
        cluster_with_freq.append((cluster_word, freq, score))

    # Сортируем по убыванию частоты
    cluster_with_freq.sort(key=lambda x: x[1], reverse=True)

    # Берём топ-20
    cluster_with_freq = cluster_with_freq[:20]

    # Рассчитываем статистику
    scores = [s for _, _, s in cluster_with_freq]
    n = len(scores)
    scores_sorted = sorted(scores)

    stats = ClusterStats(
        count=n,
        max_score=max(scores),
        mean_score=sum(scores) / n if n > 0 else 0.0,
        median_score=(scores_sorted[n // 2 - 1] + scores_sorted[n // 2]) / 2 if n % 2 == 0 else scores_sorted[n // 2],
        min_score=min(scores),
    )

    return ClusterResponse(
        word=word,
        cluster=[
            ClusterWord(word=w, freq=f, score=s)
            for w, f, s in cluster_with_freq
        ],
        stats=stats,
    )


@app.get("/api/exclusive", response_model=ExclusiveClusterResponse)
async def get_exclusive_clustering():
    """
    Получить ВСЕ результаты непересекающейся кластеризации.
    """
    if not state.loaded or state.engine is None:
        raise HTTPException(status_code=503, detail="Service not ready. Please wait for startup.")

    # Получаем ВСЕ кластеры (из кэша)
    all_clusters = await get_or_compute_all_clusters()

    # Сортируем по размеру
    sorted_clusters = sorted(all_clusters.items(), key=lambda x: len(x[1]), reverse=True)

    # Формируем ответ — отдаём всё
    cluster_list = [
        ExclusiveClusterItem(word=word, freq=len(indices))
        for word, indices in sorted_clusters
    ]

    return ExclusiveClusterResponse(clusters=cluster_list)


@app.post("/api/exclusive/iterative", response_model=ExclusiveClusterResponse)
async def get_iterative_exclusive_clustering(request: IterativeRequest):
    """
    Получить результаты итеративной непересекающейся кластеризации.
    Использует LRU кэш для производительности.
    """
    if not state.loaded or state.engine is None:
        raise HTTPException(status_code=503, detail="Service not ready. Please wait for startup.")

    seed_words = [w.lower().strip() for w in request.seed_words if w.strip()]

    if not seed_words:
        raise HTTPException(status_code=400, detail="seed_words cannot be empty")

    # Используем LRU кэш
    clusters = get_cached_iterative_clusters(seed_words)

    # Сортируем по размеру
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)

    # Формируем ответ
    cluster_list = [
        ExclusiveClusterItem(word=word, freq=len(indices))
        for word, indices in sorted_clusters
    ]

    return ExclusiveClusterResponse(clusters=cluster_list)


async def get_or_compute_all_clusters() -> Dict[str, Set[int]]:
    """
    Получить все кластеры (из кэша или вычислить).
    Кэширует результат для последующих вызовов.
    """
    if state.all_clusters is not None:
        return state.all_clusters

    if state.engine is None:
        raise HTTPException(status_code=503, detail="Service not ready. Please wait for startup.")

    print("🔄 Computing all exclusive clusters (first call)...")
    all_clusters = state.engine._exclusive_clusterer.cluster()
    state.all_clusters = all_clusters
    print(f"✅ Computed {len(all_clusters)} exclusive clusters")

    return all_clusters


def get_cached_iterative_clusters(seed_words: List[str]) -> Dict[str, Set[int]]:
    """
    Получить кластеры из LRU кэша или вычислить.
    При попадании в кэш — обновляет позицию (делает "самым свежим").
    """
    seed_words_tuple = tuple(w.lower().strip() for w in seed_words)

    # Проверяем кэш
    if seed_words_tuple in state.cluster_cache:
        # Перемещаем в конец (самый свежий)
        state.cluster_cache_order.remove(seed_words_tuple)
        state.cluster_cache_order.append(seed_words_tuple)
        print(f"📦 Cache hit for seed_words: {seed_words}")
        return state.cluster_cache[seed_words_tuple]

    # Вычисляем
    if state.engine is None:
        raise HTTPException(status_code=503, detail="Service not ready.")

    print(f"🔄 Computing clusters for seed_words: {seed_words}")
    clusters = state.engine.iterative_exclusive_clustering(seed_words=list(seed_words_tuple))

    # Добавляем в кэш с LRU eviction
    if len(state.cluster_cache_order) >= state.MAX_CACHE_SIZE:
        oldest = state.cluster_cache_order.pop(0)
        del state.cluster_cache[oldest]
        print(f"🗑️ Evicted from cache: {oldest}")

    state.cluster_cache[seed_words_tuple] = clusters
    state.cluster_cache_order.append(seed_words_tuple)
    print(f"✅ Cached clusters for {seed_words} (cache size: {len(state.cluster_cache)})")

    return clusters


@app.get("/api/exclusive/sentences", response_model=SentencesResponse)
async def get_exclusive_sentences(word: str, limit: int = 20, offset: int = 0, seed_words: str = ""):
    """
    Получить оригинальные предложения для заданного слова.
    seed_words (JSON array) — контекст итеративной кластеризации.

    Если seed_words указаны, вычисляем кластеры для подмножества предложений,
    а не для всех. Это обеспечивает консистентность с итеративным просмотром.
    """
    if not state.loaded or state.engine is None or state.data_store is None:
        raise HTTPException(status_code=503, detail="Service not ready. Please wait for startup.")

    word_lower = word.lower().strip()
    if not word_lower:
        raise HTTPException(status_code=400, detail="Word cannot be empty")

    # Определяем, использовать ли глобальные кластеры или итеративные
    import json
    seed_words_list = []
    if seed_words and seed_words.strip():
        try:
            seed_words_list = json.loads(seed_words)
        except json.JSONDecodeError:
            seed_words_list = []

    if seed_words_list:
        # Итеративная кластеризация для подмножества (с LRU кэшем)
        clusters = get_cached_iterative_clusters(seed_words_list)
    else:
        # Глобальные кластеры (из кэша или вычисляем)
        clusters = await get_or_compute_all_clusters()

    # Находим кластер для слова
    if word_lower not in clusters:
        return SentencesResponse(word=word_lower, total_count=0, sentences=[])

    sentence_indices = clusters[word_lower]
    total_count = len(sentence_indices)

    # Сортируем индексы и применяем пагинацию
    sorted_indices = sorted(sentence_indices)
    paginated_indices = sorted_indices[offset:offset + limit]

    # Получаем ОРИГИНАЛЬНЫЕ предложения (не обработанные)
    original_sentences = state.data_store.get_original_sentences_by_index(paginated_indices)

    # Полный путь поиска: seed_words + текущее слово
    search_path = seed_words_list + [word_lower]

    # Находим позиции ВСЕХ слов из пути в каждом предложении
    sentences_with_highlights = []
    for idx, text in zip(paginated_indices, original_sentences):
        positions_dict = {}
        text_lower = text.lower()

        for search_word in search_path:
            word_positions = []
            start = 0
            while True:
                pos = text_lower.find(search_word, start)
                if pos == -1:
                    break
                word_positions.append(pos)
                start = pos + 1

            if word_positions:
                positions_dict[search_word] = word_positions

        sentences_with_highlights.append(
            SentenceItem(
                index=idx,
                text=text,
                highlight_positions=positions_dict,
                search_path=search_path,
            )
        )

    return SentencesResponse(
        word=word_lower,
        total_count=total_count,
        sentences=sentences_with_highlights,
    )


@app.get("/api/health")
async def health_check():
    """Проверка готовности сервиса."""
    return {
        "status": "ok" if state.loaded else "loading",
        "engine_loaded": state.loaded,
    }


# ============================================
# PMI Sequential Clustering Endpoints
# ============================================

@app.post("/api/pmi/cluster", response_model=PmiClusterResponse)
async def get_pmi_cluster(request: PmiClusterRequest):
    """
    Получить PMI-кластер для слова.
    Если sentence_indices указаны — вычисляем PMI в подмножестве предложений.
    """
    if not state.loaded or state.engine is None:
        raise HTTPException(status_code=503, detail="Service not ready. Please wait for startup.")

    word = request.word.lower().strip()
    if not word:
        raise HTTPException(status_code=400, detail="Word cannot be empty")

    engine = state.engine

    # Если указаны индексы предложений — создаём временный SearchEngine для подмножества
    if request.sentence_indices and len(request.sentence_indices) > 0:
        sorted_indices = sorted(request.sentence_indices)
        subset_sentences = [state.data_store.get_processed_sentences()[i] for i in sorted_indices]

        temp_engine = SearchEngine(
            sentences=subset_sentences,
            calc_word_freq=True,
            enable_cluster_analysis=True,
            enable_exclusive_clustering=False,
        )

        cluster_analyzer = temp_engine._cluster_analyzer
    else:
        temp_engine = engine
        cluster_analyzer = engine._cluster_analyzer

    if cluster_analyzer is None:
        raise HTTPException(status_code=503, detail="Cluster analysis not available.")

    # Получаем PMI-кластер
    # Для подмножества используем min_score_percent=0, т.к. PMI на малых
    # подмножествах даёт низкие значения, и жёсткий порог отсекает всё
    effective_min_score_percent = 0.0 if request.sentence_indices else (request.min_score_percent or 30.0)

    cluster = cluster_analyzer.get_cluster_words(
        seed_word=word,
        top_n=20 * 5,
        use_npmi=False,
        min_freq=request.min_freq or 1,
        tfidf_range=None,
        use_freq_weighting=True,
        min_score_percent=effective_min_score_percent,
    )

    if not cluster:
        return PmiClusterResponse(
            word=word,
            cluster=[],
            stats=ClusterStats(
                count=0,
                max_score=0.0,
                mean_score=0.0,
                median_score=0.0,
                min_score=0.0,
            ),
            sentence_indices=request.sentence_indices,
        )

    # Добавляем частоту к каждому слову
    cluster_with_freq = []
    for cluster_word, score in cluster:
        freq = cluster_analyzer.word_doc_freq.get(cluster_word.lower(), 0)
        cluster_with_freq.append((cluster_word, freq, score))

    # Сортируем по убыванию частоты
    cluster_with_freq.sort(key=lambda x: x[1], reverse=True)

    # Берём топ-20
    cluster_with_freq = cluster_with_freq[:20]

    # Рассчитываем статистику
    scores = [s for _, _, s in cluster_with_freq]
    n = len(scores)
    scores_sorted = sorted(scores)

    stats = ClusterStats(
        count=n,
        max_score=max(scores),
        mean_score=sum(scores) / n if n > 0 else 0.0,
        median_score=(scores_sorted[n // 2 - 1] + scores_sorted[n // 2]) / 2 if n % 2 == 0 else scores_sorted[n // 2],
        min_score=min(scores),
    )

    return PmiClusterResponse(
        word=word,
        cluster=[
            PmiClusterWord(word=w, freq=f, score=s)
            for w, f, s in cluster_with_freq
        ],
        stats=stats,
        sentence_indices=request.sentence_indices,
    )


@app.post("/api/pmi/sentences", response_model=SentencesResponse)
async def get_pmi_sentences(request: PmiSentencesRequest):
    """
    Получить предложения для слова в контексте seed_words.
    Находим предложения, где встречаются все seed_words, и возвращаем их с подсветкой.
    """
    if not state.loaded or state.data_store is None:
        raise HTTPException(status_code=503, detail="Service not ready. Please wait for startup.")

    word = request.word.lower().strip()
    if not word:
        raise HTTPException(status_code=400, detail="Word cannot be empty")

    # Формируем путь поиска: seed_words + текущее слово
    search_path = request.seed_words + [word]

    # Находим предложения, где встречаются все слова из search_path
    all_sentences = state.data_store.get_processed_sentences()
    matching_indices = _find_sentences_with_all_words(search_path)

    total_count = len(matching_indices)

    if total_count == 0:
        return SentencesResponse(
            word=word,
            total_count=0,
            sentences=[],
        )

    # Пагинация
    sorted_indices = sorted(matching_indices)
    paginated_indices = sorted_indices[request.offset:request.offset + request.limit]

    # Получаем оригинальные предложения
    original_sentences = state.data_store.get_original_sentences_by_index(paginated_indices)

    # Находим позиции всех слов из пути для подсветки
    sentences_with_highlights = []
    for idx, text in zip(paginated_indices, original_sentences):
        positions_dict = {}
        text_lower = text.lower()

        for search_word in search_path:
            word_positions = []
            start = 0
            while True:
                pos = text_lower.find(search_word, start)
                if pos == -1:
                    break
                word_positions.append(pos)
                start = pos + 1

            if word_positions:
                positions_dict[search_word] = word_positions

        sentences_with_highlights.append(
            SentenceItem(
                index=idx,
                text=text,
                highlight_positions=positions_dict,
                search_path=search_path,
            )
        )

    return SentencesResponse(
        word=word,
        total_count=total_count,
        sentences=sentences_with_highlights,
    )


def _find_sentences_with_all_words(words: List[str]) -> set:
    """
    Найти индексы предложений, где встречаются ВСЕ указанные слова.
    """
    all_sentences = state.data_store.get_processed_sentences()
    matching_indices = set(range(len(all_sentences)))

    for word in words:
        word_lower = word.lower()
        new_indices = set()
        for idx in matching_indices:
            sent_words = all_sentences[idx].lower().split()
            if word_lower in sent_words:
                new_indices.add(idx)
        matching_indices = new_indices
        if not matching_indices:
            break

    return matching_indices


@app.post("/api/pmi/indices", response_model=Dict[str, object])
async def get_pmi_indices(request: IterativeRequest):
    """
    Получить индексы предложений, где встречаются все seed_words.
    Используется для последовательного PMI — передаём индексы для вычисления в подмножестве.
    """
    if not state.loaded or state.data_store is None:
        raise HTTPException(status_code=503, detail="Service not ready.")

    seed_words = [w.lower().strip() for w in request.seed_words if w.strip()]
    if not seed_words:
        raise HTTPException(status_code=400, detail="seed_words cannot be empty")

    matching_indices = _find_sentences_with_all_words(seed_words)

    return {
        "indices": sorted(matching_indices),
        "count": len(matching_indices),
        "seed_words": seed_words,
    }
