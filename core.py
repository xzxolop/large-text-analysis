import nltk
import kagglehub
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import streamlit as st

def make_map_most_popular(search_words, documents):
    """Ищет слова, которые встречаются вместе с заданными словами"""
    m = {}
    
    for doc in documents[:100]:
        words = nltk.word_tokenize(doc)
        
        # Проверяем, содержатся ли ВСЕ слова из поискового запроса в документе
        if all(sw in words for sw in search_words):
            for word in words:
                # Исключаем слова из поискового запроса и стоп-слова
                if (word not in search_words and 
                    word.isalnum() and 
                    word not in stopwords.words('english')):
                    if word in m:
                        m[word] = m[word] + 1
                    else:
                        m[word] = 1
    return m

@st.cache_data
def load_data():
    print("start load")
    path = Path(kagglehub.dataset_download("pavellexyr/the-reddit-dataset-dataset"))
    path_to_comments = path.joinpath("the-reddit-dataset-dataset-comments.csv")

    comments_df = pd.read_csv(path_to_comments)
    comments_body = comments_df["body"]

    df = pd.DataFrame()
    df['document'] = comments_body

    nltk.download('punkt')
    nltk.download('stopwords')

    stop_words = set(stopwords.words('english'))

    def preprocess(text):
        if not isinstance(text, str):
            return ''

        tokens = word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        return ' '.join(filtered_tokens)

    df['processed'] = df['document'].apply(preprocess)
    print("end load")
    return df

def search_word_func():
    # Используем current_search вместо прямого доступа к text_input
    search_query = st.session_state.get('current_search', '').strip()
    if not search_query:
        # Если current_search пуст, пробуем взять из text_input
        search_query = st.session_state.get('text_input', '').strip()
        st.session_state['current_search'] = search_query
    
    df = st.session_state['text_df']
    
    if search_query:
        search_words = search_query.split()
        
        # Обновляем историю поиска
        if not st.session_state['search_history'] or (st.session_state['search_history'] and st.session_state['search_history'][-1] != search_words):
            st.session_state['search_history'].append(search_words)
        
        m = make_map_most_popular(search_words, df['processed'])
        sorted_items = sorted(m.items(), key=lambda x: x[1], reverse=True)    

        def create_list(words, size: int):
            if size < 0:
                size = len(words)
            word_list = []
            count_list = []
            for i in range(0, min(size, len(words))):
                word_list.append(words[i][0])
                count_list.append(words[i][1])
            d = {'word': word_list, 'count': count_list}           
            return pd.DataFrame(data=d)
        
        result_df = create_list(sorted_items, 20)  # Ограничиваем до 20 результатов для наглядности
        st.session_state['data_frame'] = result_df
    else:
        st.session_state['data_frame'] = pd.DataFrame()
        st.session_state['search_history'] = []