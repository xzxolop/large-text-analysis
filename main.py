import pandas as pd
import streamlit as st

import core
import invertedindex as ii

# Используем AdvancedInvertedIndex вместо базового
index = ii.AdvancedInvertedIndex()

if 'index' not in st.session_state:
    st.session_state['index'] = index

text_df = core.load_data()

if 'text_df' not in st.session_state:
    st.session_state['text_df'] = text_df
    
if 'words_view_df' not in st.session_state:
    st.session_state['words_view_df'] = pd.DataFrame()

if 'sentances_view_df' not in st.session_state:
    st.session_state['sentances_view_df'] = pd.DataFrame()

st.title('Word Co-occurrence Finder')
st.write('Это приложение позволяет находить слова, которые чаще всего встречаются вместе с заданным словом. '
         'Поиск проводится на датасете the-reddit-dataset-dataset-comments.')

col1, col2 = st.columns([1, 1])
col1.text_input('Слово для поиска', key='search_word')
col2.number_input('Количество слов для показа', 
                 min_value=1, 
                 max_value=50, 
                 value=10, 
                 key='max_words')

st.button(label='Найти связанные слова', on_click=core.search_word)

st.subheader("Слова, часто встречающиеся вместе с запросом")
st.dataframe(st.session_state['words_view_df'])

st.subheader("Предложения, содержащие исходное слово")
st.dataframe(st.session_state['sentances_view_df'])