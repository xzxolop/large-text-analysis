import nltk
import kagglehub
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
import streamlit as st

def print_searched_words(words, count):
    if len(words) < count or count < 0:
        count = len(words)

    for key, value in words:
        if count < 0:
            break
        print(f"{key}: {value}")
        count -= 1    

@st.cache_data
def load_data():
    print("start load")
    path = Path(kagglehub.dataset_download("pavellexyr/the-reddit-dataset-dataset"))
    path_to_comments = path.joinpath("the-reddit-dataset-dataset-comments.csv")

    # Загрузка только нужной колонки для экономии памяти
    comments_df = pd.read_csv(path_to_comments, usecols=["body"])
    comments_body = comments_df["body"]

    all_sentences = []
    
    # Более эффективная обработка предложений
    for comment in comments_body.dropna():  # автоматически пропускаем NaN
        if isinstance(comment, str):
            sentences = sent_tokenize(comment)
            all_sentences.extend(sentences)

    # Создаем DataFrame с правильными индексами
    df = pd.DataFrame({'document': all_sentences})
    
    # Загружаем NLTK данные один раз
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

    stop_words = set(stopwords.words('english'))

    def preprocess(text):
        tokens = word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        return ' '.join(filtered_tokens)

    df['processed'] = df['document'].apply(preprocess)
    
    # Используем AdvancedInvertedIndex
    index = st.session_state['index']
    index.add_documents_serias(df['processed'])
    
    print(f"end load. Loaded {len(df)} sentences")
    return df

def search_word():
    search_word = st.session_state['search_word'].lower().strip()
    max_words = st.session_state.get('max_words', '10')
    
    try:
        max_words = int(max_words)
    except ValueError:
        max_words = 10
    
    if not search_word:
        st.warning("Введите слово для поиска")
        return
    
    index = st.session_state['index']
    text_df = st.session_state['text_df']
    
    # Поиск слов, которые часто встречаются вместе с заданным
    co_occurring_words = index.find_co_occurring_words(search_word, top_k=max_words)
    
    # Создаем DataFrame для отображения слов
    if co_occurring_words:
        words_df = pd.DataFrame(
            co_occurring_words, 
            columns=['Word', 'TF-IDF Score']
        )
        words_df['Rank'] = range(1, len(words_df) + 1)
        words_df = words_df[['Rank', 'Word', 'TF-IDF Score']]
    else:
        words_df = pd.DataFrame(columns=['Rank', 'Word', 'TF-IDF Score'])
    
    # Получаем предложения, содержащие исходное слово
    target_docs = index.get_documents_with_word(search_word)
    sentences_data = []
    
    for doc_info in target_docs[:20]:  # Ограничиваем количество предложений
        original_text = text_df.loc[doc_info['doc_id'], 'document']
        sentences_data.append({
            'Document ID': doc_info['doc_id'],
            'Sentence': original_text,
            'TF': f"{doc_info['tf']:.4f}"
        })
    
    sentences_df = pd.DataFrame(sentences_data)
    
    # Обновляем состояние
    st.session_state['words_view_df'] = words_df
    st.session_state['sentances_view_df'] = sentences_df
    
    # Дополнительная статистика
    word_stats = index.get_word_statistics(search_word)
    if word_stats:
        st.sidebar.subheader("Статистика слова")
        st.sidebar.write(f"Частота в документах: {word_stats['document_frequency']}")
        st.sidebar.write(f"IDF: {word_stats['idf']:.4f}")
        st.sidebar.write(f"Средний TF: {word_stats['average_tf']:.4f}")