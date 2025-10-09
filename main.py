import kagglehub
import pandas as pd
import nltk

import core
from utils import count_word_matches

# Note: Необходимо выполнить при первом запуске
path = kagglehub.dataset_download("pavellexyr/the-reddit-dataset-dataset")
nltk.download('punkt_tab')

path = "C:/Users/evoro/.cache/kagglehub/datasets/pavellexyr/the-reddit-dataset-dataset/versions/1"
print("Path to dataset files:", path)

posts_df = pd.read_csv(path + "\\the-reddit-dataset-dataset-posts.csv")
comments_df = pd.read_csv(path + "\\the-reddit-dataset-dataset-comments.csv")
comments_body = comments_df["body"]

sentences = core.tokenize_sentances(comments_body)
words = core.tokenize_words(sentences)

tf = core.calculate_TF("of", sentences[0])
print("tf:", tf )

idf = core.claculate_IDF("of", sentences)
print("idf:", idf)

search_word = "data"
cnt_words = count_word_matches(search_word, words)
print(f"Колличество слов {search_word}: {cnt_words}")