"""
Веб-демо для large-text-analysis с использованием FastAPI.

Запуск:
    uv run uvicorn src.web_app:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List
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
    """Хранит инициализированный SearchEngine."""
    engine: Optional[SearchEngine] = None
    loaded: bool = False


state = AppState()


@app.on_event("startup")
async def startup_event():
    """Загружает данные и инициализирует SearchEngine при старте приложения."""
    print("🔄 Loading dataset and initializing SearchEngine...")

    data_store = DataStorage()
    data_store.load_data()  # Загрузка Reddit dataset
    sentences = data_store.get_processed_sentences()

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
    Получить результаты непересекающейся кластеризации.
    """
    if not state.loaded or state.engine is None:
        raise HTTPException(status_code=503, detail="Service not ready. Please wait for startup.")

    engine = state.engine
    
    # Получаем топ-50 кластеров
    clusters = engine.get_top_exclusive_clusters(top_n=50, min_cluster_size=1)
    
    # Сортируем по размеру
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Формируем ответ
    cluster_list = [
        ExclusiveClusterItem(word=word, freq=len(indices))
        for word, indices in sorted_clusters
    ]
    
    return ExclusiveClusterResponse(clusters=cluster_list)


@app.post("/api/exclusive/iterative", response_model=ExclusiveClusterResponse)
async def get_iterative_exclusive_clustering(request: IterativeRequest):
    """
    Получить результаты итеративной непересекающейся кластеризации.
    """
    if not state.loaded or state.engine is None:
        raise HTTPException(status_code=503, detail="Service not ready. Please wait for startup.")

    engine = state.engine
    seed_words = [w.lower().strip() for w in request.seed_words if w.strip()]
    
    if not seed_words:
        raise HTTPException(status_code=400, detail="seed_words cannot be empty")
    
    # Получаем итеративную кластеризацию
    clusters = engine.iterative_exclusive_clustering(seed_words=seed_words)
    
    # Сортируем по размеру
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Формируем ответ
    cluster_list = [
        ExclusiveClusterItem(word=word, freq=len(indices))
        for word, indices in sorted_clusters
    ]
    
    return ExclusiveClusterResponse(clusters=cluster_list)


@app.get("/api/health")
async def health_check():
    """Проверка готовности сервиса."""
    return {
        "status": "ok" if state.loaded else "loading",
        "engine_loaded": state.loaded,
    }
