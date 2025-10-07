import kagglehub
import pandas as pd
import nltk

# Download latest version
path = kagglehub.dataset_download("pavellexyr/the-reddit-dataset-dataset")
print("Path to dataset files:", path)

posts_df = pd.read_csv(path + "\\the-reddit-dataset-dataset-posts.csv")
comments_df = pd.read_csv(path + "\\the-reddit-dataset-dataset-comments.csv")

#print(posts_df)
#print(comments_df)

comments_body = comments_df["body"]
s1 = comments_body[0]
print(s1)
sentances = nltk.sent_tokenize(s1)
print(sentances)

tokens = nltk.word_tokenize(s1)
print(tokens)