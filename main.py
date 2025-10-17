import kagglehub
import pandas as pd
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import core
import streamlit as st

print("start")

# Инициализация session_state
if 'searched_word' not in st.session_state:
    st.session_state.searched_word = ""
if 'word_df' not in st.session_state:
    st.session_state.word_df = pd.DataFrame({"word": [], "count": []})

def search_word(): 
    print("searched_word:", st.session_state.searched_word)
    
    # Очищаем списки перед каждым поиском
    words_list = []
    quantity_list = []
    
    m = core.make_map_most_popular(st.session_state.searched_word, df['processed'])
    sorted_items = sorted(m.items(), key=lambda x: x[1], reverse=True)    
    core.print_searched_words(sorted_items, 10)

    for key, value in sorted_items:
        words_list.append(key)
        quantity_list.append(value)

    print(words_list)
    print(quantity_list)

    # Сохраняем результат в session_state
    st.session_state.word_df = pd.DataFrame({"word": words_list, "count": quantity_list})
    print(st.session_state.word_df)

# Загрузка и предобработка данных ДО создания интерфейса
@st.cache_data
def load_and_preprocess_data():
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
    print("preprocess end")
    return df

# Загружаем данные
df = load_and_preprocess_data()

# Интерфейс
st.title("My prilo")

searched_word = st.text_input(
    label="Search word",
    on_change=search_word,
    key="searched_word"
)

st.text(body=f"search word: {searched_word}")

# Отображаем DataFrame из session_state
st.dataframe(st.session_state.word_df)