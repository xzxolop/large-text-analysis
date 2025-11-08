from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from .core import load_data, create_inverted_index

app = FastAPI(title="Word Finder API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

text_df = None
inverted_index = None

@app.on_event("startup")
async def startup_event():
    global text_df, inverted_index
    print("Загрузка данных...")
    text_df = load_data()
    print("Создание инвертированного индекса...")
    inverted_index = create_inverted_index(text_df)
    print("Готово к работе!")

class SearchRequest(BaseModel):
    search_words: List[str]  # Теперь принимаем список слов
    max_words: Optional[int] = 10

class SentenceResponse(BaseModel):
    original: str
    processed: str

class WordNode(BaseModel):
    word: str
    count: int
    children: List[Dict[str, Any]] = []  # Рекурсивная структура

class SearchResponse(BaseModel):
    word_tree: List[WordNode]
    sentences: List[SentenceResponse]

@app.get("/")
async def root():
    return {"message": "Word Finder API"}

@app.post("/search", response_model=SearchResponse)
async def search_words(request: SearchRequest):
    try:
        if not inverted_index:
            raise HTTPException(status_code=500, detail="Index not initialized")
        
        limit_words = request.max_words if request.max_words and request.max_words > 0 else -1
        
        print(f"Поиск слов: {request.search_words}, лимит: {limit_words}")
        
        # Если список слов пустой, ищем самые частые слова
        if not request.search_words:
            # Здесь можно вернуть самые частые слова из индекса
            word_tree = []
            sentences = []
        else:
            # Получаем совместно встречающиеся слова для всех слов в запросе
            word_frequency_map = inverted_index.get_co_occurring_words_multiple(
                request.search_words, 
                limit_words
            )
            
            # Строим дерево слов
            word_tree = []
            for word, count in word_frequency_map:
                word_tree.append({
                    "word": word,
                    "count": count,
                    "children": []  # Дети будут заполняться на фронтенде при раскрытии
                })
            
            # Находим предложения
            co_occurring_words = [word for word, freq in word_frequency_map]
            sentences = inverted_index.search_sentences_with_multiple_words(
                request.search_words + co_occurring_words
            )
        
        return SearchResponse(word_tree=word_tree, sentences=sentences[:50])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "documents_loaded": inverted_index.doc_count if inverted_index else 0}