import nltk
import kagglehub
from pathlib import Path
import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

class DataStorage:
    """
    Хранит данные датасета в списках python.
    __main_text_list        - список документов (текстов) датасета.\n
    __orig_sent_list        - список оригинальных предложений, получается из разбиения документов датасета на предложения.\n
    __processed_sent_list   - список обработаных предложений (равно числу оригинальных.\n
    __alias_list            - список связей, где index соответствует оригинальному и обработаному предожению, а value - документу датасета.\n
    """
    __main_text_list = []  #TODO: rename to documents_list
    __orig_sent_list = []
    __processed_sent_list = []
    __alias_list = []

    __stop_words = set()
    __url_pattern = re.compile(r'https?://\S+|www\.\S+')


    def load_data(self):
        """Загружает данные из датасета в списки python."""

        path = kagglehub.dataset_download("pavellexyr/the-reddit-dataset-dataset")
        dataset_path = Path(path) / "the-reddit-dataset-dataset-comments.csv"
        
        comments_df = pd.read_csv(dataset_path)
        self.__main_text_list = comments_df["body"].to_list()

        nltk.download('stopwords', quiet=True)
        #nltk.download('punkt', quiet=True)
        # TODO: также в качестве стоп-слов  добавить ссылки (через рег. выражения)
        self.__stop_words = set(stopwords.words('english'))
        self.__fill_lists_by_main_text()

    def load_text(self, text):
        """Эта функция позволяет вместо загрузки датасета, передать строку, которая и будет исходным текстом. Функция по-большей степени нужна для тестирования."""
        self.__main_text_list = text
        nltk.download('stopwords', quiet=True)
        self.__stop_words = set(stopwords.words('english'))
        self.__fill_lists_by_main_text()

    def get_processed_sentences(self) -> list:
        return self.__processed_sent_list
    
    def get_original_sentences_by_index(self, indexes: set) -> list:
        sent_list = []
        for i in indexes:
            sent_list.append(self.__orig_sent_list[i])
        return sent_list
    
    def get_processed_sentences_by_index(self, indexes: set) -> list:
        sent_list = []
        for i in indexes:
            sent_list.append(self.__processed_sent_list[i])
        return sent_list

    def set_stopwords():
        return
        
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
        # 1. Удаляем ссылки
        sent = re.sub(self.__url_pattern, '', sent)

        # 2. Токенизация
        words = word_tokenize(sent)

        # 3. Фильтрация
        filtered_words = [
            word.lower()
            for word in words
            if word.lower() not in self.__stop_words
            and word.isalnum()
        ]

        return " ".join(filtered_words)
