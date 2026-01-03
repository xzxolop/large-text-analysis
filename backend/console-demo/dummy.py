import nltk
import kagglehub
from pathlib import Path
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

def load_data():
    path = kagglehub.dataset_download("pavellexyr/the-reddit-dataset-dataset")
    path_to_comments = Path(path) / "the-reddit-dataset-dataset-comments.csv"
    print(f"Датасет загружен в: {path_to_comments}\n")

    comments_df = pd.read_csv(path_to_comments)
    print("comments_df:\n", comments_df, "\n")

    sentances_df = pd.DataFrame({
        "raw": comments_df["body"]  # Тут создается копия!
    })

    print("sentances_df:\n", sentances_df, "\n")

    nltk.download('stopwords', quiet=True)
    
    # TODO: также в качестве стоп-слов  добавить ссылки (через рег. выражения)
    stop_words = set(stopwords.words('english'))
    print("stop words:\n", stop_words, "\n")


    sentances_df["processed"] = ""
    print("sentances_df:\n", sentances_df, "\n")

    print("sentances list:\n")
    preprocess_df(sentances_df, stop_words)
    print(sentances_df)

    # Комментарий -> предложения -> удалить ненужные слова.
def preprocess_df(sentances_df, stop_words):
    print("size:", sentances_df["raw"].size)
    for i in range(sentances_df["raw"].size): #sentances_df["raw"].size
        sent = sentances_df["raw"][i]
        sent_list = preprocess_sent(sent, stop_words)
        sentances_df.at[i, "processed"] = sent_list

def preprocess_sent(text: str, stop_words):
    
    # Разбиваем комментарий на предложения
    filtered_setances_list = []

    if not isinstance(text, str):
            return filtered_setances_list

    setances_list = nltk.sent_tokenize(text)
    
    for sent in setances_list:
        filtered_sent = ""
        words = nltk.word_tokenize(sent)
        for word in words:
            if word not in stop_words:
                filtered_sent += word
                filtered_sent += " "
        filtered_setances_list.append(filtered_sent)
    return filtered_setances_list