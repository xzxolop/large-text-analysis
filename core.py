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
        filtered_tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(filtered_tokens)

    df['processed'] = df['document'].apply(preprocess)

    sentance_list = []
    for doc in df['processed'].values :
        lst = sent_tokenize(doc)
        for sent in lst:
            sentance_list.append(sent)

    sentances_df = pd.DataFrame()
    sentances_df['processed'] = sentance_list
    print("end load")
    return sentances_df

# Создает словарь вида <слово, колличество> внем находятся слова с которыми поисковое слово встречается наиболее часто
def make_word_frequency_map(search_word: str, sentences):
    word_frequency_map = {}
    for sent in sentences:
        words = nltk.word_tokenize(sent)
        if search_word in words:
            for word in words:
                if word != search_word and word.isalnum():
                    if word in word_frequency_map:
                        word_frequency_map[word] = (word_frequency_map[word] + 1)
                    else:
                        word_frequency_map[word] = 1
    return word_frequency_map


def word_map_to_df(word_frequency_map, limit:int):
        if limit < 0:
            limit = len(word_frequency_map)
        word_list = []
        count_list = []
        for i in range(0, min(limit, len(word_frequency_map))):
            word_list.append(word_frequency_map[i][0])
            count_list.append(word_frequency_map[i][1])
        d = {'word': word_list, 'count': count_list}           
        df = pd.DataFrame(data=d)
        return df

#search_word
def search_word():
    search_word = st.session_state['search_word'] # Зависимость от модуля более высокого урвня = нарушение DIP
    text_df = st.session_state['text_df']

    word_map = make_word_frequency_map(search_word, text_df['processed'])
    sorted_word_map = sorted(word_map.items(), key=lambda x: x[1], reverse=True)    

    limit_words = st.session_state['max_words']

    if limit_words.isdigit():
        limit_words = int(limit_words)
    else:
        limit_words = -1
    
    print(f"limit words: {limit_words}")
    words_view_df = word_map_to_df(sorted_word_map, limit_words)
    st.session_state['words_view_df'] = words_view_df 

    search_sentences_by_word()

def search_sentences_by_word():
    text_df = st.session_state['text_df']
    search_word = st.session_state['search_word']

    sentences = text_df['processed']
    words_view_df =  st.session_state['words_view_df']
    view_words = words_view_df['word'].values

    sentences_list = []
    for word in view_words:
        for sent in sentences:
            words = nltk.word_tokenize(sent)
            if word in words and search_word in words:
                sentences_list.append(sent)
    
    st.session_state['sentances_view_df'] = pd.DataFrame(sentences_list)
    
    #return sentences_list



