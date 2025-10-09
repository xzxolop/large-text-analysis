import kagglehub
import pandas as pd
import nltk
from pathlib import Path

import core
from utils import count_word_matches

# Note: Необходимо выполнить при первом запуске
nltk.download('punkt_tab')

path = Path(kagglehub.dataset_download("pavellexyr/the-reddit-dataset-dataset"))
path_to_posts = path.joinpath("the-reddit-dataset-dataset-posts.csv")
path_to_comments = path.joinpath("the-reddit-dataset-dataset-comments.csv")

posts_df = pd.read_csv(path_to_posts)
comments_df = pd.read_csv(path_to_comments)
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