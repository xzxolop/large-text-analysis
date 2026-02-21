import nltk
import kagglehub
from pathlib import Path
import pandas as pd
import re
import os

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

    def load_data(self):
        """Загружает данные из датасета в списки python."""

        path = kagglehub.dataset_download("pavellexyr/the-reddit-dataset-dataset")
        dataset_path = Path(path) / "the-reddit-dataset-dataset-comments.csv"
        
        comments_df = pd.read_csv(dataset_path)
        self.__main_text_list = comments_df["body"].to_list()

        nltk.download('stopwords', quiet=True)
        self.__stop_words = set(stopwords.words('english'))
        self.__fill_lists_by_main_text()

    def writeProcessedTextToFile(self, filename="output.txt"):
        """Сохраняет элементы __processed_text_list в текстовый файл."""
        
        filepath = os.path.join("files", filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            text_lines = [str(item) for item in self.__processed_sent_list]
            f.write('\n'.join(text_lines))
        
        print(f"Сохранено {len(self.__processed_sent_list)} строк в {filepath}")

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
        sent_without_links = self.__delete_links(sent)

        words = word_tokenize(sent_without_links)
        filtered_words = [word for word in words 
                            if word not in self.__stop_words and word.isalnum()]
        return " ".join(filtered_words)
    
    def __delete_links(self, text):
        """Очищает текст от ссылок."""
        if pd.isna(text):
            return ""
        
        # Преобразуем в строку
        text = str(text)
        
        # Удаляем URL (http/https/ftp)
        url_pattern = r'https?://\S+|ftp://\S+'
        text = re.sub(url_pattern, '', text)
        
        # Удаляем URL без протокола (начинающиеся с www.)
        www_pattern = r'www\.\S+'
        text = re.sub(www_pattern, '', text)
        
        # Удаляем URL в скобках
        bracket_pattern = r'\(https?://\S+\)|\(www\.\S+\)'
        text = re.sub(bracket_pattern, '', text)
        
        # Удаляем URL в квадратных скобках
        square_bracket_pattern = r'\[https?://\S+\]|\[www\.\S+\]'
        text = re.sub(square_bracket_pattern, '', text)
        
        # Удаляем URL в угловых скобках
        angle_bracket_pattern = r'<https?://\S+>|<www\.\S+>'
        text = re.sub(angle_bracket_pattern, '', text)
        
        # Удаляем ссылки формата [text](url)
        markdown_pattern = r'\[.*?\]\(https?://\S+\)'
        text = re.sub(markdown_pattern, '', text)
        
        return text