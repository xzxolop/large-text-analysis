import pandas as pd
import streamlit as st

import core

text_df = core.load_data()

st.dataframe(text_df)

# Инициализация session state
if 'text_df' not in st.session_state:
    st.session_state['text_df'] = text_df

if 'inverted_index' not in st.session_state:
    st.session_state['inverted_index'] = core.create_inverted_index(text_df['processed'].tolist())

if 'words_view_df' not in st.session_state:
    st.session_state['words_view_df'] = pd.DataFrame()

if 'sentances_view_df' not in st.session_state:
    st.session_state['sentances_view_df'] = pd.DataFrame()

st.title('Word finder')
st.write('Это приложение позволяет проводить поиск слов, которые наиболее часто встречаются в тексте.' \
' Поиск проводится на датасете the-reddit-dataset-dataset-comments.')

col1, col2 = st.columns([1, 1])
col1.text_input('Слово для поиска', key='search_word')
col2.text_input('Ограничение по количеству слов', key='max_words')

st.button(label='Поиск', on_click=core.search_word)

col3, col4 = st.columns([1,2])
col3.dataframe(st.session_state['words_view_df'])
col4.dataframe(st.session_state['sentances_view_df'])