import nltk
import kagglehub
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import streamlit as st

# Создает словарь вида <слово, колличество> внем находятся слова с которыми поисковое слово встречается наиболее часто
#TODO : сделать возвращ. значение DF
def make_word_frequency_map(search_word: str, sentences):
    word_frequency_map = {}
    for doc in sentences:
        words = nltk.word_tokenize(doc)
        if search_word in words:
            for word in words:
                if word != search_word:
                    if word in word_frequency_map:
                        word_frequency_map[word] = (word_frequency_map[word] + 1)
                    else:
                        word_frequency_map[word] = 1
    return word_frequency_map

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
    word = st.session_state['text_input'] # Зависимость от модуля более высокого урвня = нарушение DIP
    df = st.session_state['text_df']
    m = make_word_frequency_map(word, df['processed'])
    sorted_items = sorted(m.items(), key=lambda x: x[1], reverse=True)    

    def create_list(words, size: int):
        if size < 0:
            size = len(words)
        word_list = []
        count_list = []
        for i in range(0, min(size, len(words))):  # Добавлена проверка на размер
            word_list.append(words[i][0])
            count_list.append(words[i][1])
        d = {'word': word_list, 'count': count_list}           
        df = pd.DataFrame(data=d)
        return df
    
    df = create_list(sorted_items, -1)
    st.session_state['data_frame'] = df 