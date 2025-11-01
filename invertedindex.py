import nltk
import kagglehub
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
import streamlit as st

import invertedindex as ii

# Глобальная переменная для индекса (кешируем)
@st.cache_resource
def create_inverted_index(_texts):
    """Создает и возвращает инвертированный индекс"""
    print("Создание инвертированного индекса...")
    index = ii.InvertedIndex()
    index.add_documents(_texts)
    print(f"Индекс создан! Слов: {len(index.index)}, Документов: {index.doc_count}")
    return index

@st.cache_data
def load_data():
    print("start load")
    path = Path(kagglehub.dataset_download("pavellexyr/the-reddit-dataset-dataset"))
    path_to_comments = path.joinpath("the-reddit-dataset-dataset-comments.csv")

    comments_df = pd.read_csv(path_to_comments)
    comments_body = comments_df["body"]

    all_sentences = []

    # Проходим по каждому комментарию
    for comment in comments_body:
        if isinstance(comment, str):  # проверяем, что это строка
            sentences = sent_tokenize(comment)
            for sent in sentences:
                all_sentences.append(sent)  # добавляем все предложения в общий список

    df = pd.DataFrame()
    df['document'] = all_sentences

    nltk.download('punkt')
    nltk.download('stopwords')

    stop_words = set(stopwords.words('english'))

    def preprocess(text):
        if not isinstance(text, str):
            return ''

        # Сначала разбиваем на предложения
        sentences = sent_tokenize(text)
        all_filtered_tokens = []
        
        for sentence in sentences:
            # Токенизируем каждое предложение отдельно
            tokens = word_tokenize(sentence.lower())
            filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
            all_filtered_tokens.extend(filtered_tokens)
        
        # Возвращаем все токены как один текст
        return ' '.join(all_filtered_tokens)

    df['processed'] = df['document'].apply(preprocess)
    print("end load")
    return df

def word_map_to_df(word_frequency_map, limit:int):
    """Конвертирует словарь частот в DataFrame"""
    if limit < 0:
        limit = len(word_frequency_map)
    
    word_list = []
    count_list = []
    
    for i in range(min(limit, len(word_frequency_map))):
        word_list.append(word_frequency_map[i][0])
        count_list.append(word_frequency_map[i][1])
    
    df = pd.DataFrame({'word': word_list, 'count': count_list})
    return df

def search_word():
    """Основная функция поиска с использованием инвертированного индекса"""
    search_word = st.session_state['search_word']
    text_df = st.session_state['text_df']
    
    # Получаем инвертированный индекс (кешируется)
    inverted_index = st.session_state['inverted_index']
    
    # Определяем лимит слов
    try:
        limit_words = int(st.session_state['max_words'])
    except:
        limit_words = -1
    
    print(f"Поиск слова: {search_word}, лимит: {limit_words}")
    
    # Используем инвертированный индекс для поиска совместно встречающихся слов
    word_frequency_map = inverted_index.get_co_occurring_words(search_word, limit_words)
    
    # Создаем DataFrame для отображения
    words_view_df = word_map_to_df(word_frequency_map, limit_words)
    st.session_state['words_view_df'] = words_view_df
    
    # Находим предложения с этими словами
    co_occurring_words = [word for word, freq in word_frequency_map]
    sentences = inverted_index.search_sentences_with_words(search_word, co_occurring_words)
    
    # Ограничиваем количество предложений для производительности
    sentences = sentences[:50]  # максимум 50 предложений
    
    st.session_state['sentances_view_df'] = pd.DataFrame(sentences, columns=['sentence'])