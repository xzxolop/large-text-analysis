import nltk
import kagglehub
from pathlib import Path
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

class DataStorage:
    __main_text_list = [] #TODO: rename to documents_list
    __orig_sent_list = []
    __processed_sent_list = []
    __alias_list = []
    __dataset_path = ""

    __stop_words = set()

    def get_sentances(self):
        return self.__processed_sent_list
    
    def get_dataset_path(self):
        return self.__dataset_path

    def set_stopwords():
        return

    def load_data(self):
        path = kagglehub.dataset_download("pavellexyr/the-reddit-dataset-dataset")
        self.__dataset_path = Path(path) / "the-reddit-dataset-dataset-comments.csv"
        
        comments_df = pd.read_csv(self.__dataset_path)
        self.__main_text_list = comments_df["body"].to_list()

        nltk.download('stopwords', quiet=True)
        # TODO: также в качестве стоп-слов  добавить ссылки (через рег. выражения)
        self.__stop_words = set(stopwords.words('english'))

        self.__fill_lists_by_main_text()
    
    def get_original_sentences(self, indexes: set) -> list:
        sent_list = []
        for i in indexes:
            sent_list.append(self.__orig_sent_list[i])
        return sent_list
        
    def __fill_lists_by_main_text(self):
        for i in range(len(self.__main_text_list)):
            text = self.__main_text_list[i]
            if not isinstance(text, str):
                self.__orig_sent_list.append("")
                self.__processed_sent_list.append("")
                self.__alias_list.append(i)
            else:
                sent_list = sent_tokenize(text)
                for sent in sent_list:
                    self.__orig_sent_list.append(sent)
                    proc_sent = self.__preprocess_sent(sent)
                    self.__processed_sent_list.append(proc_sent)
                    self.__alias_list.append(i)

    def __preprocess_sent(self, sent: str):
        proc_sent = ""
        words = word_tokenize(sent)
        for word in words:
            if word not in self.__stop_words:
                proc_sent += word
                proc_sent += " "
        return proc_sent   