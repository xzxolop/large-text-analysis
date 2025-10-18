import kagglehub
import pandas as pd
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import streamlit as st

import core

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
    word = st.session_state['text_input']
    df = st.session_state['text_df']
    m = core.make_map_most_popular(word, df['processed'])

    sorted_items = sorted(m.items(), key=lambda x: x[1], reverse=True)    

    core.print_searched_words(sorted_items, 10)

    def create_list(words, size: int):
        word_list = []
        count_list = []
        for i in range(0, size+1):
            word_list.append(words[i][0])
            count_list.append(words[i][1])
        d = {'word':word_list, 'count':count_list}           
        df = pd.DataFrame(data=d)
        return df
    
    df = create_list(sorted_items, 10)
    print(df)

    if 'data_frame' in st.session_state:
        print('is yet!')
        print(st.session_state['data_frame'])
    else:
        print('not yet!')

    st.session_state['data_frame'] = df 


text_df = load_data()

if 'text_df' not in st.session_state:
    st.session_state['text_df'] = text_df
    
if 'data_frame' not in st.session_state:
    st.session_state['data_frame'] = pd.DataFrame()

st.title('My prilo')
st.text_input('Search word', key='text_input', on_change=search_word_func)
st.dataframe(st.session_state['data_frame'])