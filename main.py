import kagglehub
import pandas as pd
import nltk
from utils import print_arr, count_word_matches

# Download latest version
# path = kagglehub.dataset_download("pavellexyr/the-reddit-dataset-dataset")
path = "C:/Users/evoro/.cache/kagglehub/datasets/pavellexyr/the-reddit-dataset-dataset/versions/1"
print("Path to dataset files:", path)

posts_df = pd.read_csv(path + "\\the-reddit-dataset-dataset-posts.csv")
comments_df = pd.read_csv(path + "\\the-reddit-dataset-dataset-comments.csv")

#print(posts_df)
#print(comments_df)

comments_body = comments_df["body"]

sentences = []

for x in comments_body:
    if isinstance(x, str):
        sentences += nltk.sent_tokenize(x)

#print_arr(sentences, "sentences")

words = []

for x in sentences:
    words += nltk.word_tokenize(x)

search_word = "data"

cnt_words = count_word_matches(search_word, words)
print(cnt_words)