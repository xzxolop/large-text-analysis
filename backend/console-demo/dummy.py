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

    # Комментарии
    print("comments body:\n", comments_df["body"], "\n")

    nltk.download('stopwords', quiet=True)
    
    # TODO: также в качестве стоп-слов  добавить ссылки (через рег. выражения)
    stop_words = set(stopwords.words('english'))
    print("stop words:\n", stop_words, "\n")

    print("sentances list:\n")
    for i in range(1):
        sent = preprocess(comments_df["body"][i], stop_words)
        print(f"After preprocess:\n {sent}")

    # Комментарий -> предложения -> удалить ненужные слова.
def preprocess(text: str, stop_words):
    
    # Разбиваем комментарий на предложения
    setances_list = nltk.sent_tokenize(text)
    print(setances_list, "\n")

    filtered_setances_list = []
    
    for sent in setances_list:
        filtered_sent = ""
        words = nltk.word_tokenize(sent)
        print(f"words: \n{words}")
        for word in words:
            if word not in stop_words:
                filtered_sent += word
                filtered_sent += " "
        filtered_setances_list.append(filtered_sent)
    return filtered_setances_list
        
                
        

    
