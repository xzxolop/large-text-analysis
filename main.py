import kagglehub
import pandas as pd
from pathlib import Path

import core

path = Path(kagglehub.dataset_download("pavellexyr/the-reddit-dataset-dataset"))
path_to_comments = path.joinpath("the-reddit-dataset-dataset-comments.csv")

comments_df = pd.read_csv(path_to_comments)
comments_body = comments_df["body"]

df = pd.DataFrame()
df['document'] = comments_body
print(df)

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('russian'))

def preprocess(text):
    if not isinstance(text, str):
        return ''

    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)

df['processed'] = df['document'].apply(preprocess)
print(df)

# Применение tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['processed'])

# Вывод TF-IDF в виде массивов
print(tfidf_matrix.toarray())
print(vectorizer.get_feature_names_out())

# Вывод в виде таблицы
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
print(tfidf_df)

cnt = 0
# Вывод в виде каждого документа отдельно
# TODO: сделать вывод наиболее важных if value > 0
for index, row in tfidf_df.iterrows():
    print(f"Документ {index + 1}:")
    print(row.sort_values(ascending=False).head(5)) # выводит самые важные, но значение константно
    cnt += 1
    if (cnt == 100):
        break
