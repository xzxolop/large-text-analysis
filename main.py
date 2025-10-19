
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
st.text_input('Search word', key='text_input', on_change=core.search_word_func)
st.dataframe(st.session_state['data_frame'])