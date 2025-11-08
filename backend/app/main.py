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
    search_words: List[str]
    max_words: Optional[int] = 10

class SentenceResponse(BaseModel):
    original: str

class SearchResponse(BaseModel):
    word_tree: List[Dict[str, Any]]
    sentences: List[SentenceResponse]

@app.get("/")
async def root():
    return {"message": "Word Finder API"}

@app.post("/search", response_model=SearchResponse)
async def search_words(request: SearchRequest):
    try:
        if not inverted_index:
            raise HTTPException(status_code=500, detail="Index not initialized")
        
        limit_words = request.max_words if request.max_words and request.max_words > 0 else 10
        
        print(f"Поиск слов: {request.search_words}, лимит: {limit_words}")
        
        # Получаем совместно встречающиеся слова
        word_frequency_map = inverted_index.get_co_occurring_words_multiple(
            request.search_words, 
            limit_words
        )
        
        # Строим дерево слов
        word_tree = []
        for word, count in word_frequency_map:
            word_tree.append({
                "word": word,
                "count": count
            })
        
        # Находим предложения для текущего набора слов
        sentences = inverted_index.search_sentences_with_multiple_words(request.search_words)
        
        return SearchResponse(
            word_tree=word_tree, 
            sentences=[SentenceResponse(original=sent) for sent in sentences[:20]]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "documents_loaded": inverted_index.doc_count if inverted_index else 0}