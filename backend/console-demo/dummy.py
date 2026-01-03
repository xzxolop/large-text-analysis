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

    # NOTE: На мой взгляд, иметь отдельный df для хранения оригинальных и обработанных предложений - хорошоая идея
    sentances_df = pd.DataFrame({
        "raw": comments_df["body"]  # Тут создается копия!
    })
    sentances_df["processed"] = ""

    print("sentances_df:\n", sentances_df, "\n")

    nltk.download('stopwords', quiet=True)
    
    # TODO: также в качестве стоп-слов  добавить ссылки (через рег. выражения)
    stop_words = set(stopwords.words('english'))
    print("stop words:\n", stop_words, "\n")

    print("sentances list:\n")
    preprocess_df(sentances_df, stop_words)
    print(sentances_df)

def preprocess_df(sentances_df, stop_words):
    for i in range(sentances_df["raw"].size):
        sent = sentances_df["raw"][i]
        sent_list = preprocess_sent(sent, stop_words)
        sentances_df.at[i, "processed"] = sent_list

def preprocess_sent(text: str, stop_words):
    filtered_setances_list = []

    if not isinstance(text, str):
            return filtered_setances_list

    setances_list = sent_tokenize(text)
    
    for sent in setances_list:
        filtered_sent = ""
        words = word_tokenize(sent)
        for word in words:
            if word not in stop_words:
                filtered_sent += word
                filtered_sent += " "
        filtered_setances_list.append(filtered_sent)
    return filtered_setances_list



#########################
# Вариант 2. "4 списка"
#########################

# Слишком много копирования кода
def load_dataV2():
    path = kagglehub.dataset_download("pavellexyr/the-reddit-dataset-dataset")
    path_to_comments = Path(path) / "the-reddit-dataset-dataset-comments.csv"
    print(f"Датасет загружен в: {path_to_comments}\n")

    comments_df = pd.read_csv(path_to_comments)
    print("comments_df:\n", comments_df, "\n")

    text_list = []
    orig_list = []
    proc_list = []
    alias_list = []

    text_list = comments_df["body"].to_list()

    nltk.download('stopwords', quiet=True)
    
    # TODO: также в качестве стоп-слов  добавить ссылки (через рег. выражения)
    stop_words = set(stopwords.words('english'))
    print("stop words:\n", stop_words, "\n")

    text_to_sent(text_list, orig_list, proc_list, stop_words)
    print(orig_list[:5])
    print(proc_list[:5])
    

def text_to_sent(text_list: list, orig_list: list, proc_list: list, stop_words):
    
    
    for text in text_list:
        # TODO: тут делать связку
        if not isinstance(text, str):
            orig_list.append("")
            proc_list.append("")
        else:
            sent_list = sent_tokenize(text)
            for sent in sent_list:
                orig_list.append(sent)
                proc_sent = preprocess_sentV2(sent, stop_words)
                proc_list.append(proc_sent)

def preprocess_sentV2(sent: str, stop_words):
    proc_sent = ""
    words = word_tokenize(sent)
    for word in words:
        if word not in stop_words:
            proc_sent += word
            proc_sent += " "
    return proc_sent         



