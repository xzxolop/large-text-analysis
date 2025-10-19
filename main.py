
import pandas as pd
import streamlit as st

import core

text_df = core.load_data()

if 'text_df' not in st.session_state:
    st.session_state['text_df'] = text_df
    
if 'data_frame' not in st.session_state:
    st.session_state['data_frame'] = pd.DataFrame()

st.title('Word finder')
st.write('Это приложение позволяет проводить поиск слов, которые наиболее часто встречаются в тексте.' \
' Поиск проводится на датасете the-reddit-dataset-dataset-comments.')

col1, col2 = st.columns([1, 1])
col1.text_input('Слово для поиска', key='search_word')
col2.text_input('Ограничение по колличеству слов', key='max_words')

st.button(label='Поиск', on_click=core.search_word)

st.dataframe(st.session_state['data_frame'])