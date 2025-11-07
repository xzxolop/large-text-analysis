from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Исправленные импорты
from .core import load_data, create_inverted_index

app = FastAPI(title="Word Finder API")

# CORS для работы с фронтендом
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене укажите конкретный домен
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные
text_df = None
inverted_index = None

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    global text_df, inverted_index
    print("Загрузка данных...")
    text_df = load_data()
    print("Создание инвертированного индекса...")
    inverted_index = create_inverted_index(text_df['processed'].tolist())
    print("Готово к работе!")

class SearchRequest(BaseModel):
    search_word: str
    max_words: Optional[int] = 10

class SearchResponse(BaseModel):
    words: List[dict]
    sentences: List[str]

@app.get("/")
async def root():
    return {"message": "Word Finder API"}

@app.post("/search", response_model=SearchResponse)
async def search_words(request: SearchRequest):
    try:
        if not inverted_index:
            raise HTTPException(status_code=500, detail="Index not initialized")
        
        # Определяем лимит слов
        limit_words = request.max_words if request.max_words and request.max_words > 0 else -1
        
        print(f"Поиск слова: '{request.search_word}', лимит: {limit_words}")
        
        # Используем инвертированный индекс для поиска
        word_frequency_map = inverted_index.get_co_occurring_words(
            request.search_word, 
            limit_words
        )
        
        # Форматируем слова для ответа
        words_data = [{"word": word, "count": count} for word, count in word_frequency_map]
        
        # Находим предложения
        co_occurring_words = [word for word, freq in word_frequency_map]
        sentences = inverted_index.search_sentences_with_words(
            request.search_word, 
            co_occurring_words
        )
        
        return SearchResponse(words=words_data, sentences=sentences[:50])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "documents_loaded": inverted_index.doc_count if inverted_index else 0}