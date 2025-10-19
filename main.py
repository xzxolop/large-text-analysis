import pandas as pd
import streamlit as st

import core

text_df = core.load_data()

if 'text_df' not in st.session_state:
    st.session_state['text_df'] = text_df
    
if 'data_frame' not in st.session_state:
    st.session_state['data_frame'] = pd.DataFrame()

if 'search_history' not in st.session_state:
    st.session_state['search_history'] = []

if 'current_search' not in st.session_state:
    st.session_state['current_search'] = ""

st.title('Word finder')
st.write('Это приложение позволяет проводить поиск слов, которые наиболее часто встречаются в тексте.' \
' Поиск проводится на датасете the-reddit-dataset-dataset-comments.')

# Текущий контекст поиска
if st.session_state['search_history']:
    current_search_display = " + ".join(st.session_state['search_history'][-1]) if st.session_state['search_history'] else ""
    st.info(f"Текущий поиск: {current_search_display}")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Очистить поиск"):
            st.session_state['search_history'] = []
            st.session_state['data_frame'] = pd.DataFrame()
            st.session_state['current_search'] = ""
            st.rerun()
    with col2:
        if len(st.session_state['search_history']) > 1:
            if st.button("Назад"):
                st.session_state['search_history'].pop()
                if st.session_state['search_history']:
                    st.session_state['current_search'] = " ".join(st.session_state['search_history'][-1])
                else:
                    st.session_state['current_search'] = ""
                core.search_word_func()
                st.rerun()

# Форма для поиска с использованием st.form
with st.form("search_form"):
    search_input = st.text_input('Search word', 
                                value=st.session_state['current_search'], 
                                key='text_input')
    submitted = st.form_submit_button("Поиск", on_click=core.search_word_func)

def handle_word_click(parent_words, new_word):
    """Обработчик клика по слову - создает новый уровень поиска"""
    new_search = parent_words + [new_word]
    st.session_state['search_history'].append(new_search)
    st.session_state['current_search'] = " ".join(new_search)
    core.search_word_func()
    st.rerun()

def display_results_tree(results_df, parent_words=None, level=0):
    """Рекурсивная функция для отображения древовидной структуры"""
    if parent_words is None:
        parent_words = []
    
    for index, row in results_df.iterrows():
        word = row['word']
        count = row['count']
        
        # Отступ для вложенности
        indent = "    " * level
        
        with st.expander(f"{indent}🔍 {word} ({count})"):
            st.write(f"**Слово:** {word}")
            st.write(f"**Частота:** {count}")
            
            # Кнопка для углубления поиска
            if st.button("Искать с этим словом", 
                        key=f"search_{level}_{word}_{index}_{len(st.session_state.get('search_history', []))}"):
                handle_word_click(parent_words, word)

# Отображение основной структуры
with st.container():
    df = st.session_state['data_frame']
    
    if not df.empty:
        st.subheader("Результаты поиска:")
        
        # Получаем текущий путь поиска
        current_path = st.session_state['search_history'][-1] if st.session_state['search_history'] else []
        
        # Отображаем результаты для текущего уровня
        display_results_tree(df, current_path)
                
    elif st.session_state['current_search']:
        st.info("По вашему запросу ничего не найдено. Попробуйте другие слова.")